"""Shared definitions for the milestone 5 validations: test systems, seeded cases, measurements.

The test modules and ``experiments/stage1/lyapunov_validation.py`` both import from here, so a
gate and the number printed for it have one implementation (D-076). Every value comes from
``horizon.tolerances`` (D-079 to D-086); nothing is chosen or tuned in this module. The Lorenz
system and the linear test systems live here, not in ``src/``, because they are validation
fixtures (D-076).

Measuring and checking are separate, as in ``tests/integrator_checks.py``: a measurement returns
numbers and never raises on a bad value, so the script can print them; a ``check_*`` function
raises. A check that cannot judge (a conditioning above ``KAPPA_CAP``, a residual under the
floor, a state above its ``X`` cap) raises with the word ``inconclusive``, and never passes
(D-045).

The values the calibration lab sets (D-083) are read from ``horizon.tolerances`` when a run is
built, not at import, so this module imports before they are logged.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from horizon import integrators, lyapunov
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from tests import integrator_checks as chk

DEFAULT = dp.DEFAULT_PARAMS
EPS = float(np.finfo(np.float64).eps)
INCONCLUSIVE = "inconclusive"
PENDULUM_MASK = (True, True, False, False)
"""The angle mask in ``(theta, omega)`` and in ``(q, p)`` (D-078)."""
NO_ANGLES_3 = (False, False, False)
NO_ANGLES_2 = (False, False)
NO_ANGLES_4 = (False, False, False, False)


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


# ---- the test systems (D-080, D-081) -------------------------------------------------------


class LorenzParams(NamedTuple):
    """``(sigma, rho, beta)`` of the Lorenz system, a JAX pytree."""

    sigma: float
    rho: float
    beta: float


LORENZ = LorenzParams(tol.LORENZ_SIGMA, tol.LORENZ_RHO, tol.LORENZ_BETA)
LORENZ_CONTROL = LorenzParams(
    tol.LORENZ_SIGMA, tol.LORENZ_RHO, tol.LORENZ_BETA * tol.LORENZ_CONTROL_BETA_FACTOR
)
"""The control of D-079: ``beta`` changed in the field, not in ``ell``."""


def lorenz(y: Array, params: LorenzParams) -> Array:
    """The Lorenz field ``(sigma (y - x), x (rho - z) - y, x y - beta z)``."""
    sigma, rho, beta = params
    return jnp.stack([sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]])


def linear(y: Array, a: Array) -> Array:
    """The linear field ``A y``; ``params`` is the matrix ``A``."""
    return a @ y


def l4_matrix(real_eigenvalue: float = tol.L4_D[0][0]) -> np.ndarray:
    """``A = V D V^-1`` of L4 (D-080), with the real eigenvalue of ``D[0, 0]`` replaceable."""
    d = np.array(tol.L4_D)
    d[0, 0] = real_eigenvalue
    v = np.array(tol.L4_V)
    return v @ d @ np.linalg.inv(v)


def ho_matrix() -> np.ndarray:
    """``A = [[0, 1], [-Omega**2, 0]]`` of HO (D-080)."""
    return np.array([[0.0, 1.0], [-(tol.HO_OMEGA**2), 0.0]])


def l4_continuous_eigenvalues() -> np.ndarray:
    """The eigenvalues of ``D``, written from its blocks: ``0.5``, ``-0.2 +- i`` and ``-1``."""
    d = tol.L4_D
    pair = complex(d[1][1], d[1][2])
    return np.array([d[0][0], pair, pair.conjugate(), d[3][3]])


def ho_continuous_eigenvalues() -> np.ndarray:
    """``+- i Omega``."""
    return np.array([1j * tol.HO_OMEGA, -1j * tol.HO_OMEGA])


def rk4_stability(z: np.ndarray) -> np.ndarray:
    """``R(z) = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24``, RK4 on ``y' = mu y`` (section 1.3)."""
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def rk4_map_exponents(mu: np.ndarray, dt: float) -> np.ndarray:
    """The exact exponents of the RK4 map, ``ln|R(dt mu)| / dt``, sorted in decreasing order.

    Written as ``ln|1 + w| = log1p(2 Re w + |w|**2) / 2`` with ``w = R(z) - 1``, because for
    HO ``|R|`` differs from 1 by about ``y**6 / 144``, below the rounding of ``R`` itself, and
    ``w`` is summed without forming ``1 + ...``.
    """
    z = dt * np.asarray(mu, dtype=np.complex128)
    w = z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0  # R(z) - 1, without forming 1 + ...
    return np.sort(0.5 * np.log1p(2.0 * w.real + np.abs(w) ** 2) / dt)[::-1]


def unit_eigenvectors(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The complex eigenvalues and unit-column eigenvectors of ``a``, by decreasing real part.

    The order within a tie is NumPy's; the bound of section 1.2 does not depend on it for the
    gated ``k`` (section 1.4 of the note).
    """
    mu, v = np.linalg.eig(a)
    order = np.argsort(-mu.real, kind="stable")
    v = v[:, order]
    return mu[order], v / np.linalg.norm(v, axis=0)


def linear_bound(a: np.ndarray, frame: np.ndarray, k: int, horizon: float) -> float:
    """``B_k = [k ln kappa(V) + ln(vol_k(C) / |det C_K|)] / T`` (note, sections 1.1, 1.2, 1.4).

    ``C = V^-1 frame[:, :k]``. For ``k = 1`` with a tie in the leading real part, ``|c_1|`` is
    the norm of the component in the tied group (the owner's tie rule, section 1.1).
    """
    mu, v = unit_eigenvectors(a)
    kappa = float(np.linalg.cond(v, 2))
    c = np.linalg.solve(v, np.asarray(frame, dtype=np.complex128)[:, :k])
    vol = math.sqrt(abs(float(np.linalg.det(c.conj().T @ c).real)))
    if k == 1:
        tied = np.abs(mu.real - mu[0].real) <= 1e-12 * max(1.0, abs(mu[0].real))
        lead = float(np.linalg.norm(c[tied, 0]))
    else:
        lead = abs(complex(np.linalg.det(c[:k, :k])))
    return (k * math.log(kappa) + math.log(vol / lead)) / horizon


# ---- seeded cases (D-084) ------------------------------------------------------------------


def draw_frame(rng: np.random.Generator, n: int) -> np.ndarray:
    """The ``Q`` of ``np.linalg.qr(rng.standard_normal((n, n)))``, signed so ``diag(R) > 0``."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def _rng(stream: int) -> np.random.Generator:
    return np.random.default_rng([tol.CASES_SEED, stream])


def _frames(stream: int, n: int, count: int) -> np.ndarray:
    rng = _rng(stream)
    return _frozen(np.stack([draw_frame(rng, n) for _ in range(count)]))


def _lorenz_points(stream: int, count: int) -> np.ndarray:
    lo_xy, hi_xy = tol.LORENZ_XY_BOX
    lo_z, hi_z = tol.LORENZ_Z_BOX
    low = np.array([lo_xy, lo_xy, lo_z])
    high = np.array([hi_xy, hi_xy, hi_z])
    return _frozen(_rng(stream).uniform(low, high, size=(count, 3)))


@functools.cache
def l4_frame() -> np.ndarray:
    """The L4 frame, ``4 x 4``, from ``L4_FRAME_STREAM``."""
    return _frames(tol.L4_FRAME_STREAM, 4, 1)[0]


@functools.cache
def ho_frame() -> np.ndarray:
    """The HO frame, ``2 x 2``, from ``HO_FRAME_STREAM``."""
    return _frames(tol.HO_FRAME_STREAM, 2, 1)[0]


@functools.cache
def lorenz_initial_conditions() -> np.ndarray:
    """The Lorenz initial conditions, shape ``(N_LORENZ_ICS, 3)``."""
    return _lorenz_points(tol.LORENZ_IC_STREAM, tol.N_LORENZ_ICS)


@functools.cache
def lorenz_frames() -> np.ndarray:
    """One ``3 x 3`` frame per Lorenz initial condition, shape ``(N_LORENZ_ICS, 3, 3)``."""
    return _frames(tol.LORENZ_FRAME_STREAM, 3, tol.N_LORENZ_ICS)


@functools.cache
def pendulum_states() -> np.ndarray:
    """The rest states of the agreement runs, by the recipe of D-046(a), shape ``(8, 4)``."""
    rng = _rng(tol.LYAP_PENDULUM_STREAM)
    return _frozen(chk.draw_rest_states(rng, DEFAULT, tol.N_LYAP_PENDULUM))


@functools.cache
def pendulum_frames() -> np.ndarray:
    """One ``4 x 4`` frame per pendulum state, shape ``(N_LYAP_PENDULUM, 4, 4)``."""
    return _frames(tol.LYAP_PENDULUM_FRAME_STREAM, 4, tol.N_LYAP_PENDULUM)


@functools.cache
def step_pendulum_states() -> np.ndarray:
    """The ``(theta, omega)`` states of the per-step residual, shape ``(N_STEP_STATES, 4)``."""
    rng = _rng(tol.STEP_PENDULUM_STREAM)
    n = tol.N_STEP_STATES
    theta = rng.uniform(-tol.CHAOTIC_THETA_MAX, tol.CHAOTIC_THETA_MAX, size=(n, 2))
    omega = rng.uniform(-tol.MOVING_OMEGA_MAX, tol.MOVING_OMEGA_MAX, size=(n, 2))
    return _frozen(np.hstack([theta, omega]))


@functools.cache
def step_lorenz_points() -> np.ndarray:
    """The Lorenz points of the per-step residual, shape ``(N_STEP_STATES, 3)``."""
    return _lorenz_points(tol.STEP_LORENZ_STREAM, tol.N_STEP_STATES)


# ---- the per-step residual (D-079(b), D-086) -----------------------------------------------


def damped_canonical_field(z: Array, params: dp.Params) -> Array:
    """The ``(q, p)`` control of D-086: ``p' <- p' - QP_CONTROL_DAMPING * p``."""
    dz = dp.canonical_vector_field(z, params)
    return dz.at[2:].add(-tol.QP_CONTROL_DAMPING * z[2:])


def _log_det_m(theta: Array, params: dp.Params) -> Array:
    return jnp.linalg.slogdet(dp.mass_matrix(theta, params))[1]


Ell = Callable[[Array, Array, float], Array]
"""``ell(x_n, x_next, dt)``: the exact log-volume change of the flow over one step."""


def ell_lorenz(x: Array, x_next: Array, dt: float) -> Array:
    """``-(sigma + 1 + beta) dt`` at the declared parameters."""
    return jnp.asarray(-(tol.LORENZ_SIGMA + 1.0 + tol.LORENZ_BETA) * dt)


def ell_canonical(x: Array, x_next: Array, dt: float) -> Array:
    """``0``: the ``(q, p)`` flow preserves volume."""
    return jnp.asarray(0.0)


def ell_theta(x: Array, x_next: Array, dt: float) -> Array:
    """``ln det M(theta_n) - ln det M(theta_{n+1})``, D-008 for one step."""
    return _log_det_m(x[:2], DEFAULT) - _log_det_m(x_next[:2], DEFAULT)


def ell_theta_reciprocal(x: Array, x_next: Array, dt: float) -> Array:
    """The control of D-079: the reciprocal of ``ell_theta``."""
    return -ell_theta(x, x_next, dt)


def step_residual(field: lyapunov.Field, params: object, x: Array, dt: float, ell: Ell) -> float:
    """``r = ln|det J_step(x, dt)| - ell``, with ``J_step = jacfwd(rk4_step)`` (D-079(b))."""
    x = jnp.asarray(x, dtype=jnp.float64)

    def step(y: Array) -> Array:
        return integrators.rk4_step(field, y, params, dt)

    log_det = jnp.linalg.slogdet(jax.jacfwd(step)(x))[1]
    return float(log_det - ell(x, step(x), dt))


@dataclass(frozen=True)
class StepMeasurement:
    """The per-step residual over a grid of steps and a set of states (D-086)."""

    grid: tuple[float, ...]
    residuals: np.ndarray
    """Shape ``(n_states, n_grid)``."""

    @property
    def largest(self) -> np.ndarray:
        """``R(dt)``: the largest ``|r|`` over the states at each ``dt``."""
        return np.max(np.abs(self.residuals), axis=0)

    @property
    def slope(self) -> float:
        """The gated statistic: the slope of ``log R`` against ``log dt``."""
        return chk.log_log_slope(self.grid, self.largest)

    @property
    def floor_value(self) -> float:
        """``R`` at the smallest ``dt``, which the floor applies to."""
        return float(self.largest[int(np.argmin(self.grid))])

    @property
    def state_slopes(self) -> np.ndarray:
        """Reported, not gated: each state's slope of ``log|r|``."""
        return np.array([chk.log_log_slope(self.grid, np.abs(r)) for r in self.residuals])

    @property
    def state_one_sign(self) -> np.ndarray:
        """Reported, not gated: whether each state's ``r`` keeps one sign over the grid."""
        signs = np.sign(self.residuals)
        return np.all(signs == signs[:, :1], axis=1)


def measure_step(
    field: lyapunov.Field,
    params: object,
    states: np.ndarray,
    grid: tuple[float, ...],
    ell: Ell,
) -> StepMeasurement:
    """The residual of every state at every ``dt`` of the grid."""
    residuals = np.array([[step_residual(field, params, x, h, ell) for h in grid] for x in states])
    return StepMeasurement(tuple(grid), residuals)


def check_step(measurement: StepMeasurement, *, what: str) -> float:
    """The floor on ``R`` at the smallest ``dt`` (inconclusive below it), then the slope >= 4.0."""
    floor = tol.STEP_FLOOR_FACTOR * EPS
    value = measurement.floor_value
    if not (math.isfinite(value) and value >= floor):
        raise AssertionError(
            f"{what}: {INCONCLUSIVE}, R = {value!r} at the smallest dt is under the validity "
            f"floor {floor:.3e}"
        )
    return chk.check_at_least(measurement.slope, tol.STEP_SLOPE_MIN, what=f"{what}, slope")


def canonical_step_states() -> np.ndarray:
    """The per-step states in ``(q, p)``: ``to_canonical`` of ``step_pendulum_states``."""
    to_canonical = jax.vmap(dp.to_canonical, in_axes=(0, None))
    return np.asarray(to_canonical(jnp.asarray(step_pendulum_states()), DEFAULT))


# ---- the bookkeeping identity (D-079(a)) ---------------------------------------------------


def logdet_identity_ratios(result: lyapunov.QRResult, steps_per_qr: int) -> np.ndarray:
    """``|sum_i ln|R_ii| - log_det_steps| / (c eps N_Q n kappa_j)`` per interval."""
    if result.log_det_steps is None:
        raise ValueError("the run needs step_logdet=True")
    log_r = np.asarray(result.log_r)
    n = log_r.shape[1]
    kappa = np.asarray(result.cond_r)
    bound = tol.LOGDET_C * EPS * steps_per_qr * n * kappa
    return np.abs(log_r.sum(axis=1) - np.asarray(result.log_det_steps)) / bound


def check_kappa_cap(result: lyapunov.QRResult, *, what: str) -> None:
    """Every ``kappa_j <= KAPPA_CAP``, the validity rule of the bound of D-079(a); inconclusive
    otherwise."""
    kappa = np.asarray(result.cond_r)
    if not np.all(np.isfinite(kappa)) or float(np.max(kappa)) > tol.KAPPA_CAP:
        raise AssertionError(
            f"{what}: {INCONCLUSIVE}, max kappa_2(R_j) = {float(np.max(kappa)):.3e} exceeds "
            f"KAPPA_CAP = {tol.KAPPA_CAP:.0e}"
        )


def check_logdet_identity(result: lyapunov.QRResult, steps_per_qr: int, *, what: str) -> float:
    """Every ``kappa_j <= KAPPA_CAP`` (inconclusive otherwise), then every ratio at most 1."""
    check_kappa_cap(result, what=what)
    ratios = logdet_identity_ratios(result, steps_per_qr)
    worst = float(np.max(ratios)) if np.all(np.isfinite(ratios)) else math.inf
    return chk.check_at_most(worst, 1.0, what=f"{what}, identity ratio (bound = 1)")


# ---- the agreement gate (D-082, D-085) -----------------------------------------------------


def delta_bq(x: float, dt: float) -> float:
    """``delta_BQ^wc = s eps X / (d0 dt)`` (D-085)."""
    return tol.DELTA_BQ_S * EPS * x / (tol.LYAP_D0 * dt)


def delta_bq_probabilistic(x: float, dt: float, n_total: int) -> float:
    """Report only: ``z delta_BQ^wc / sqrt(N_total)`` (D-085)."""
    return tol.DELTA_BQ_Z * delta_bq(x, dt) / math.sqrt(n_total)


def pendulum_x_cap(state: np.ndarray) -> float:
    """The cap on ``X`` for a pendulum run from ``state`` in ``(theta, omega)`` (note section 6)."""
    m1, m2, l1, l2, _ = DEFAULT
    e_s = chk.energy_scale(DEFAULT)
    k_max = float(dp.energy(jnp.asarray(state), DEFAULT)) + e_s + tol.X_CAP_ENERGY_ALLOWANCE * e_s
    trace_m = (m1 + m2) * l1**2 + m2 * l2**2
    p_max = math.sqrt(2.0 * k_max * trace_m)
    omega_max = math.sqrt(2.0 * k_max * trace_m / (m1 * m2 * l1**2 * l2**2))
    t_max = tol.PENDULUM_T_TRANS + tol.LYAP_T
    return max(float(np.max(np.abs(state[:2]))) + t_max * omega_max, p_max)


def lorenz_x_cap() -> float:
    """``1.01 (sigma + rho + R)`` with the trapping-ball radius ``R`` (note section 6)."""
    sigma, rho, beta = tol.LORENZ_SIGMA, tol.LORENZ_RHO, tol.LORENZ_BETA
    radius = beta * (sigma + rho) / (2.0 * math.sqrt(beta - 1.0))
    return tol.X_CAP_FACTOR * (sigma + rho + radius)


def check_x_cap(x: float, x_cap: float, *, what: str) -> None:
    """``X <= X_cap``, the validity rule of ``delta_BQ`` (D-085); inconclusive otherwise."""
    if not (math.isfinite(x) and x <= x_cap):
        raise AssertionError(f"{what}: {INCONCLUSIVE}, X = {x!r} exceeds X_cap = {x_cap:.4g}")


def check_agreement(
    lambda_b: float, lambda_qr: float, x: float, x_cap: float, dt: float, *, what: str
) -> float:
    """``X <= X_cap`` (inconclusive otherwise), then ``|lambda_B - lambda_QR,1| <= delta_BQ``."""
    check_x_cap(x, x_cap, what=what)
    return chk.check_at_most(abs(lambda_b - lambda_qr), delta_bq(x, dt), what=what)


# ---- the seams for the controls (D-077, D-079, D-080, D-081) -------------------------------


@functools.cache
def euler_tangent_step(f: lyapunov.Field) -> Callable[..., Array]:
    """The Euler-frame control of D-079: the frame follows the ``jvp`` of an explicit Euler step."""

    def tangent(x: Array, params: object, dt: Array, v: Array) -> Array:
        return jax.jvp(lambda y: y + dt * f(y, params), (x,), (v,))[1]

    return tangent


def keep_frame(y: Array) -> tuple[Array, Array]:
    """The control of D-080 and D-081: ``R`` is recorded, the frame is never replaced by ``Q``."""
    return y, jnp.linalg.qr(y)[1]


def qr_with(
    f: lyapunov.Field,
    state0: Array,
    params: object,
    frame0: Array,
    *,
    dt: float,
    steps_per_qr: int,
    n_transient: int,
    n_qr: int,
    tangent_step: Callable[[lyapunov.Field], Callable[..., Array]] | None = None,
    orthonormalize: Callable[[Array], tuple[Array, Array]] | None = None,
    step_logdet: bool = True,
) -> lyapunov.QRResult:
    """``tangent_qr`` through its private core with a control injected (the seam of D-077)."""
    return lyapunov._tangent_qr_core(
        lyapunov._rk4_step_of(f),
        (tangent_step or lyapunov._rk4_tangent_of)(f),
        orthonormalize or lyapunov._householder,
        jnp.asarray(state0, dtype=jnp.float64),
        params,
        jnp.asarray(frame0, dtype=jnp.float64),
        dt,
        steps_per_qr,
        n_transient,
        n_qr,
        step_logdet,
    )


# ---- the runs ------------------------------------------------------------------------------


def lab_value(name: str) -> int:
    """A value the calibration lab sets (D-083), read when a run is built.

    Lab values are step counts; anything but an ``int`` raises instead of being truncated (D-089).
    """
    if not hasattr(tol, name):
        raise AttributeError(f"{name} is a lab value of D-083 and is not in the register yet")
    value = getattr(tol, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} = {value!r} is a lab value of D-083 and must be an int step count")
    return value


def steps(horizon: float, dt: float) -> int:
    """``horizon / dt`` as an exact integer count of steps."""
    count = round(horizon / dt)
    if abs(count * dt - horizon) > 1e-9 * max(1.0, horizon):
        raise ValueError(f"{dt!r} does not divide {horizon!r}")
    return count


def intervals(total_steps: int, per_interval: int) -> int:
    """``total_steps / per_interval``, which must divide exactly (D-083)."""
    if total_steps % per_interval:
        raise ValueError(f"{per_interval} does not divide {total_steps}")
    return total_steps // per_interval


@dataclass(frozen=True)
class Run:
    """The two estimators from one state and direction, over a common window (D-077)."""

    benettin: lyapunov.BenettinResult
    qr: lyapunov.QRResult
    tau_b: float
    tau_q: float

    @property
    def lambda_b(self) -> float:
        return float(lyapunov.exponents(self.benettin.log_growth, self.tau_b))

    @property
    def lambdas_qr(self) -> np.ndarray:
        return np.asarray(lyapunov.exponents(self.qr.log_r, self.tau_q))


def run_both(
    field: lyapunov.Field,
    state0: np.ndarray,
    params: object,
    frame: np.ndarray,
    *,
    dt: float,
    transient_steps: int,
    window_steps: int,
    steps_per_renorm: int,
    steps_per_qr: int,
    angle_mask: tuple[bool, ...],
    step_logdet: bool = True,
) -> Run:
    """Benettin and QR over the same window, Benettin's direction the frame's first column."""
    qr = lyapunov.tangent_qr(
        field,
        jnp.asarray(state0),
        params,
        jnp.asarray(frame),
        dt=dt,
        steps_per_qr=steps_per_qr,
        n_transient=intervals(transient_steps, steps_per_qr),
        n_qr=intervals(window_steps, steps_per_qr),
        step_logdet=step_logdet,
    )
    benettin = lyapunov.benettin(
        field,
        jnp.asarray(state0),
        params,
        jnp.asarray(frame[:, 0]),
        dt=dt,
        d0=tol.LYAP_D0,
        steps_per_renorm=steps_per_renorm,
        n_transient=intervals(transient_steps, steps_per_renorm),
        n_renorm=intervals(window_steps, steps_per_renorm),
        angle_mask=angle_mask,
    )
    return Run(benettin, qr, steps_per_renorm * dt, steps_per_qr * dt)


def linear_run(a: np.ndarray, state0: np.ndarray, frame: np.ndarray) -> Run:
    """An L4 or HO run: ``AGREE_DT``, ``T = LYAP_T``, no transient, ``tau_B = tau_Q = 1``."""
    n = len(state0)
    return run_both(
        linear,
        state0,
        jnp.asarray(a),
        frame,
        dt=tol.AGREE_DT,
        transient_steps=0,
        window_steps=steps(tol.LYAP_T, tol.AGREE_DT),
        steps_per_renorm=tol.LINEAR_STEPS_PER_INTERVAL,
        steps_per_qr=tol.LINEAR_STEPS_PER_INTERVAL,
        angle_mask=(False,) * n,
    )


@functools.cache
def l4_run() -> Run:
    """The L4 run from ``L4_STATE0`` (D-080, D-089)."""
    return linear_run(l4_matrix(), np.array(tol.L4_STATE0), l4_frame())


@functools.cache
def ho_run() -> Run:
    """The HO run from ``HO_STATE0`` (D-080)."""
    return linear_run(ho_matrix(), np.array(tol.HO_STATE0), ho_frame())


def lorenz_transient_steps() -> int:
    """The Lorenz transient in steps, the lab's value (D-083)."""
    return lab_value("LORENZ_TRANSIENT_STEPS")


@functools.cache
def lorenz_run(index: int) -> Run:
    """Both estimators on Lorenz initial condition ``index`` (D-081)."""
    return run_both(
        lorenz,
        lorenz_initial_conditions()[index],
        LORENZ,
        lorenz_frames()[index],
        dt=tol.LORENZ_DT,
        transient_steps=lorenz_transient_steps(),
        window_steps=steps(tol.LORENZ_T, tol.LORENZ_DT),
        steps_per_renorm=tol.LORENZ_STEPS_PER_RENORM,
        steps_per_qr=lab_value("LORENZ_STEPS_PER_QR"),
        angle_mask=NO_ANGLES_3,
    )


def pendulum_canonical_run(index: int, field: lyapunov.Field = dp.canonical_vector_field) -> Run:
    """Both estimators in ``(q, p)`` from pendulum state ``index`` (D-082)."""
    z0 = np.asarray(dp.to_canonical(jnp.asarray(pendulum_states()[index]), DEFAULT))
    return run_both(
        field,
        z0,
        DEFAULT,
        pendulum_frames()[index],
        dt=tol.AGREE_DT,
        transient_steps=steps(tol.PENDULUM_T_TRANS, tol.AGREE_DT),
        window_steps=steps(tol.LYAP_T, tol.AGREE_DT),
        steps_per_renorm=lab_value("PENDULUM_STEPS_PER_RENORM"),
        steps_per_qr=lab_value("PENDULUM_STEPS_PER_QR"),
        angle_mask=PENDULUM_MASK,
    )


@functools.cache
def pendulum_run(index: int) -> Run:
    """The cached ``(q, p)`` run of pendulum state ``index``."""
    return pendulum_canonical_run(index)


@functools.cache
def pendulum_theta_qr(index: int) -> lyapunov.QRResult:
    """QR in ``(theta, omega)`` from the same state, the D-008 diagnostic (D-079, D-082).

    The frame is the orthonormalized image of the ``(q, p)`` frame under ``DPhi^-1`` at the
    initial state, signed so that ``diag(R) > 0``; its first column is then the image of
    Benettin's direction, which the coordinate-mismatch control of D-082 needs.
    """
    state = jnp.asarray(pendulum_states()[index])
    jac = np.asarray(jax.jacfwd(dp.to_canonical)(state, DEFAULT))
    q, r = np.linalg.qr(np.linalg.solve(jac, pendulum_frames()[index]))
    steps_per_qr = lab_value("PENDULUM_STEPS_PER_QR")
    return lyapunov.tangent_qr(
        dp.vector_field,
        state,
        DEFAULT,
        jnp.asarray(q * np.sign(np.diag(r))),
        dt=tol.AGREE_DT,
        steps_per_qr=steps_per_qr,
        n_transient=intervals(steps(tol.PENDULUM_T_TRANS, tol.AGREE_DT), steps_per_qr),
        n_qr=intervals(steps(tol.LYAP_T, tol.AGREE_DT), steps_per_qr),
        step_logdet=True,
    )


def d008_reference(result: lyapunov.QRResult, horizon: float) -> float:
    """``(1/T) ln[det M(theta_0) / det M(theta_T)]`` over the recorded window (D-008)."""
    states = np.asarray(result.states)
    first = float(_log_det_m(jnp.asarray(states[0, :2]), DEFAULT))
    last = float(_log_det_m(jnp.asarray(states[-1, :2]), DEFAULT))
    return (first - last) / horizon
