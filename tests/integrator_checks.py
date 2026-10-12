"""Shared definitions for the milestone 3 gates: problems, seeded cases, measurements, checks.

The test modules and ``experiments/stage1/integrator_validation.py`` both import from here, so a
gate and the number printed for it have one implementation. Every value comes from
``horizon.tolerances`` (D-044 to D-048); nothing is chosen or tuned in this module. The claims
and the tests that check them are tabulated in ``docs/derivations/integrators.md``, section 8.

Measuring and checking are separate. A ``measure_*`` or ``*_errors`` function returns numbers and
never raises on a bad value, so the script can print them; a ``check_*`` function raises
``AssertionError`` on a violation, so a test can fail, and returns the margin it used.
"""

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from horizon import integrators
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp

# The D-031 recipe for one parameter set. It is private to ``tests.cases`` and reused here so
# that the wiring check cannot drift from it.
from tests.cases import _draw_params

type Trajectory = Callable[..., Array]
"""Anything with the signature of ``integrators.rk4_trajectory``."""


# ---- comparison helpers -------------------------------------------------------------------


def _require_positive_finite(value: float, name: str) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def max_abs_diff(a: object, b: object) -> float:
    """Return the largest ``|a - b|`` over all elements. A NaN in either input gives NaN."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise AssertionError(f"shape mismatch, {x.shape} versus {y.shape}")
    return float(np.max(np.abs(x - y)))


def check_at_most(value: float, ceiling: float, *, what: str) -> float:
    """Raise unless ``value <= ceiling``; return ``value / ceiling`` (at most 1 on a pass).

    NaN and infinity fail, so a statistic that went wrong cannot slip under a ceiling.
    """
    _require_positive_finite(ceiling, "ceiling")
    if not math.isfinite(value):
        raise AssertionError(f"{what}: the statistic is {value!r}, not a finite number")
    ratio = value / ceiling
    if not value <= ceiling:
        raise AssertionError(
            f"{what}: {value:.3e} exceeds the ceiling {ceiling:.3e} (ratio {ratio:.3g})"
        )
    return ratio


def check_at_least(value: float, floor: float, *, what: str) -> float:
    """Raise unless ``value >= floor``; return ``floor / value`` (at most 1 on a pass).

    The floor must be positive: a floor of zero would pass vacuously. NaN and infinity fail.
    """
    _require_positive_finite(floor, "floor")
    if not math.isfinite(value):
        raise AssertionError(f"{what}: the statistic is {value!r}, not a finite number")
    if not value >= floor:
        raise AssertionError(f"{what}: {value:.3e} is below the floor {floor:.3e}")
    return floor / value


def check_in_band(value: float, band: tuple[float, float], *, what: str) -> float:
    """Raise unless ``band[0] <= value <= band[1]``; return the fraction of the half-width used.

    The return value is 0 at the centre of the band and at most 1 on a pass. NaN and infinity
    fail.
    """
    low, high = band
    if not low < high:
        raise ValueError(f"the band must have low < high, got {band!r}")
    if not math.isfinite(value):
        raise AssertionError(f"{what}: the statistic is {value!r}, not a finite number")
    if not low <= value <= high:
        raise AssertionError(f"{what}: {value:.4g} is outside the band [{low}, {high}]")
    return abs(value - 0.5 * (low + high)) / (0.5 * (high - low))


def log_log_slope(dts: Sequence[float], errors: Sequence[float]) -> float:
    """Return the least-squares slope of ``log(errors)`` against ``log(dts)``.

    A non-positive or non-finite error has no logarithm, so the slope is NaN and the checks
    reject it.
    """
    error_array = np.asarray(errors, dtype=np.float64)
    if not np.all(np.isfinite(error_array) & (error_array > 0.0)):
        return math.nan
    return float(np.polyfit(np.log(np.asarray(dts, dtype=np.float64)), np.log(error_array), 1)[0])


def steps_per_unit(dt: float) -> int:
    """Return ``stride`` such that ``stride * dt = 1``, so outputs fall on integer times."""
    stride = round(1.0 / dt)
    if stride < 1 or abs(stride * dt - 1.0) > 1e-12:
        raise ValueError(f"dt = {dt!r} does not divide the output spacing 1")
    return stride


# ---- the four exact-solution problems (D-044) ---------------------------------------------
#
# Each ``f`` takes ``(state, params)`` like every right-hand side here; ``params`` is ``None``.
# Each ``exact`` maps a scalar time to the state, in JAX so that ``jax.jvp`` can check it against
# its own equation (X1 to X4). The closed forms are derived in section 3 of the note.

HARMONIC_Y0 = (1.0, 0.5)
LINEAR_Y0 = (1.0, -1.0)
LOGISTIC_Y0 = (0.1,)
ROTATION_Y0 = (1.0, 0.5)

LINEAR_MATRIX = np.array([[-1.0, 5.0], [0.0, -2.0]])


def f_harmonic(y: Array, params: None) -> Array:
    """``x' = v``, ``v' = -4 x``."""
    return jnp.stack([y[1], -4.0 * y[0]])


def f_linear(y: Array, params: None) -> Array:
    """``y' = A y`` with the non-normal matrix ``A = [[-1, 5], [0, -2]]``."""
    return jnp.asarray(LINEAR_MATRIX) @ y


def f_logistic(y: Array, params: None) -> Array:
    """``y' = y (1 - y)``."""
    return y * (1.0 - y)


def f_rotation(y: Array, params: None) -> Array:
    """``x' = -(x^2 + y^2) y``, ``y' = (x^2 + y^2) x``."""
    radius_squared = y[0] ** 2 + y[1] ** 2
    return jnp.stack([-radius_squared * y[1], radius_squared * y[0]])


def exact_harmonic(t: Array) -> Array:
    """``x = x0 cos 2t + (v0 / 2) sin 2t``, ``v = -2 x0 sin 2t + v0 cos 2t``."""
    x0, v0 = HARMONIC_Y0
    return jnp.stack(
        [
            x0 * jnp.cos(2.0 * t) + 0.5 * v0 * jnp.sin(2.0 * t),
            -2.0 * x0 * jnp.sin(2.0 * t) + v0 * jnp.cos(2.0 * t),
        ]
    )


def exact_linear(t: Array) -> Array:
    """``y2 = y2(0) e^-2t``, ``y1 = y1(0) e^-t + 5 y2(0) (e^-t - e^-2t)``."""
    y1_0, y2_0 = LINEAR_Y0
    decay1, decay2 = jnp.exp(-t), jnp.exp(-2.0 * t)
    return jnp.stack([y1_0 * decay1 + 5.0 * y2_0 * (decay1 - decay2), y2_0 * decay2])


def exact_logistic(t: Array) -> Array:
    """``y = y0 e^t / (1 - y0 + y0 e^t)``, which is ``1 / (1 + 9 e^-t)`` for ``y0 = 0.1``."""
    (y0,) = LOGISTIC_Y0
    growth = jnp.exp(t)
    return jnp.stack([y0 * growth / (1.0 - y0 + y0 * growth)])


def exact_rotation(t: Array) -> Array:
    """Constant radius ``r0``; the angle is ``phi0 + r0^2 t``."""
    x0, y0 = ROTATION_Y0
    radius = math.hypot(x0, y0)
    angle = math.atan2(y0, x0) + radius**2 * t
    return jnp.stack([radius * jnp.cos(angle), radius * jnp.sin(angle)])


class ExactProblem(NamedTuple):
    """One exact-solution problem with the horizon and ceiling declared for it (D-044)."""

    name: str
    f: Callable[[Array, None], Array]
    exact: Callable[[Array], Array]
    y0: np.ndarray
    horizon: int
    ceiling: float


_EXACT_DEFINITIONS = {
    "harmonic": (f_harmonic, exact_harmonic, HARMONIC_Y0),
    "linear": (f_linear, exact_linear, LINEAR_Y0),
    "logistic": (f_logistic, exact_logistic, LOGISTIC_Y0),
    "rotation": (f_rotation, exact_rotation, ROTATION_Y0),
}


@functools.cache
def exact_problems() -> tuple[ExactProblem, ...]:
    """Return the problems of ``tol.EXACT_GATES``, in the order of the register."""
    problems = []
    for gate in tol.EXACT_GATES:
        f, exact, y0 = _EXACT_DEFINITIONS[gate.name]
        problems.append(
            ExactProblem(
                gate.name, f, exact, np.array(y0, dtype=np.float64), gate.horizon, gate.ceiling
            )
        )
    return tuple(problems)


def exact_rows(problem: ExactProblem) -> np.ndarray:
    """Return the exact solution at ``t = 0, 1, ..., horizon``, shape ``(horizon + 1, n)``."""
    times = jnp.arange(problem.horizon + 1, dtype=jnp.float64)
    return np.asarray(jax.vmap(problem.exact)(times))


def rk4_exact_errors(problem: ExactProblem, trajectory: Trajectory | None = None) -> np.ndarray:
    """Return RK4's worst-case error against the exact solution for each ``dt`` of the grid.

    The statistic is the maximum over the output times ``t = 0, 1, ..., horizon`` and the
    components. ``trajectory`` defaults to ``integrators.rk4_trajectory``; the negative controls
    pass a deliberately wrong scheme with the same signature.
    """
    run = integrators.rk4_trajectory if trajectory is None else trajectory
    exact = exact_rows(problem)
    errors = []
    for dt in tol.EXACT_DT_GRID:
        rows = run(
            problem.f, jnp.asarray(problem.y0), None, dt, problem.horizon, steps_per_unit(dt)
        )
        errors.append(max_abs_diff(rows, exact))
    return np.array(errors)


def wrapper_exact_error(problem: ExactProblem) -> float:
    """Return the DOP853 wrapper's worst-case error against the exact solution.

    The wrapper runs at its default tolerances, as in D-044.
    """
    times = np.arange(problem.horizon + 1, dtype=np.float64)
    rows = integrators.solve_reference(problem.f, problem.y0, None, times)
    return max_abs_diff(rows, exact_rows(problem))


def check_exact_rk4(problem: ExactProblem, errors: np.ndarray) -> tuple[float, float]:
    """The RK4 half of the D-044 gate: the slope in its band and the finest step under its ceiling.

    Returns the margins used by the two checks.
    """
    slope = log_log_slope(tol.EXACT_DT_GRID, errors)
    used = check_in_band(
        slope, tol.EXACT_SLOPE_BAND, what=f"{problem.name}: slope of log(error) against log(dt)"
    )
    finest = int(np.argmin(tol.EXACT_DT_GRID))
    ratio = check_at_most(
        float(errors[finest]),
        problem.ceiling,
        what=f"{problem.name}: error at dt = {tol.EXACT_DT_GRID[finest]}",
    )
    return used, ratio


# ---- seeded pendulum cases (D-045 to D-047) -----------------------------------------------

_energy = jax.jit(dp.energy)


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def draw_rest_states(rng: np.random.Generator, params: dp.Params, n: int) -> np.ndarray:
    """Draw ``n`` states at rest with positive energy under ``params``.

    Per attempt ``rng.uniform(-pi, pi, size=2)`` gives the angles, ``omega = 0``, and the state is
    kept only if ``E > 0``. This is "chaotic-ish", not a classification (milestone 6).
    """
    kept = []
    while len(kept) < n:
        theta = rng.uniform(-tol.CHAOTIC_THETA_MAX, tol.CHAOTIC_THETA_MAX, size=2)
        state = np.array([theta[0], theta[1], 0.0, 0.0])
        if float(_energy(jnp.asarray(state), params)) > 0.0:
            kept.append(state)
    return np.array(kept)


@functools.cache
def regular_states() -> np.ndarray:
    """Return the regular states, shape ``(N_REGULAR_STATES, 4)``, at rest (D-045)."""
    rng = np.random.default_rng([tol.CASES_SEED, tol.CONV_STREAM])
    theta = rng.uniform(
        -tol.REGULAR_THETA_MAX, tol.REGULAR_THETA_MAX, size=(tol.N_REGULAR_STATES, 2)
    )
    return _frozen(np.hstack([theta, np.zeros_like(theta)]))


@functools.cache
def chaotic_states() -> np.ndarray:
    """Return the chaotic-ish states for the default parameters, shape ``(N_CHAOTIC_STATES, 4)``."""
    rng = np.random.default_rng([tol.CASES_SEED, tol.CHAOTIC_STREAM])
    return _frozen(draw_rest_states(rng, dp.DEFAULT_PARAMS, tol.N_CHAOTIC_STATES))


def energy_scale(params: dp.Params) -> float:
    """Return ``E_s = g ((m1 + m2) L1 + m2 L2)``, the largest possible ``|V|`` (D-047)."""
    return params.g * ((params.m1 + params.m2) * params.L1 + params.m2 * params.L2)


@dataclass(frozen=True)
class WiringCase:
    """One non-default parameter set with one state drawn for it (D-047)."""

    index: int
    params: dp.Params
    state: np.ndarray
    t_s: float
    """``sqrt(L1 / g)`` of this set, the time unit of its gate."""


@functools.cache
def wiring_cases() -> tuple[WiringCase, ...]:
    """Return the ``N_WIRE_SETS * N_WIRE_STATES`` cases, set by set in the declared order."""
    param_rng = np.random.default_rng([tol.CASES_SEED, tol.WIRE_PARAM_STREAM])
    state_rng = np.random.default_rng([tol.CASES_SEED, tol.WIRE_STATE_STREAM])
    sets = [dp.Params(*_draw_params(param_rng)) for _ in range(tol.N_WIRE_SETS)]
    cases = []
    for params in sets:
        for state in draw_rest_states(state_rng, params, tol.N_WIRE_STATES):
            cases.append(
                WiringCase(len(cases), params, _frozen(state), math.sqrt(params.L1 / params.g))
            )
    return tuple(cases)


# ---- measurements on the pendulum ---------------------------------------------------------


def rk4_pendulum(
    states: np.ndarray,
    params: dp.Params,
    dt: float,
    n_out: int,
    stride: int,
    trajectory: Trajectory | None = None,
) -> np.ndarray:
    """Run RK4 from each state with common parameters; shape ``(n_states, n_out + 1, 4)``."""
    run = integrators.rk4_trajectory if trajectory is None else trajectory
    batched = jax.vmap(lambda y0: run(dp.vector_field, y0, params, dt, n_out, stride))
    return np.asarray(batched(jnp.asarray(states)))


def reference_pendulum(
    states: np.ndarray, params: dp.Params, times: np.ndarray, reference_tol: float
) -> np.ndarray:
    """Run the DOP853 wrapper from each state; shape ``(n_states, len(times), 4)``."""
    return np.stack(
        [
            integrators.solve_reference(
                dp.vector_field, state, params, times, rtol=reference_tol, atol=reference_tol
            )
            for state in states
        ]
    )


@dataclass(frozen=True)
class Convergence:
    """Pendulum convergence measurements per regular state (D-045)."""

    errors: np.ndarray
    """Shape ``(n_states, n_dt)``: RK4's worst-case error against the ``1e-13`` reference."""
    self_errors: np.ndarray
    """Shape ``(n_states,)``: distance between the ``1e-13`` and ``2.5e-14`` references."""
    slopes: np.ndarray
    """Shape ``(n_states,)``: least-squares slope of ``log(error)`` against ``log(dt)``."""


def measure_convergence() -> Convergence:
    """Measure the convergence gate of D-045 on the regular states, default parameters."""
    states = regular_states()
    times = np.arange(tol.CONV_T + 1, dtype=np.float64)
    reference = reference_pendulum(states, dp.DEFAULT_PARAMS, times, tol.CONV_REFERENCE_TOL)
    check_reference = reference_pendulum(states, dp.DEFAULT_PARAMS, times, tol.CONV_SELF_CHECK_TOL)
    self_errors = np.max(np.abs(reference - check_reference), axis=(1, 2))
    errors = np.empty((len(states), len(tol.CONV_DT_GRID)))
    for j, dt in enumerate(tol.CONV_DT_GRID):
        rows = rk4_pendulum(states, dp.DEFAULT_PARAMS, dt, tol.CONV_T, steps_per_unit(dt))
        errors[:, j] = np.max(np.abs(rows - reference), axis=(1, 2))
    slopes = np.array([log_log_slope(tol.CONV_DT_GRID, row) for row in errors])
    return Convergence(errors, self_errors, slopes)


def agreement_errors(states: np.ndarray, horizon: int, reference_tol: float) -> np.ndarray:
    """Return ``max over components of |RK4 - DOP853|``, shape ``(n_states, horizon + 1)``.

    Default parameters, ``dt = AGREE_DT``, outputs at ``t = 0, 1, ..., horizon``. The gates
    take the maximum of this array (D-046); the script prints its median and maximum per time.
    """
    rk4 = rk4_pendulum(
        states, dp.DEFAULT_PARAMS, tol.AGREE_DT, horizon, steps_per_unit(tol.AGREE_DT)
    )
    times = np.arange(horizon + 1, dtype=np.float64)
    reference = reference_pendulum(states, dp.DEFAULT_PARAMS, times, reference_tol)
    return np.max(np.abs(rk4 - reference), axis=2)


def chaotic_agreement_errors() -> np.ndarray:
    """The chaotic-ish agreement measurement of D-046(a)."""
    return agreement_errors(chaotic_states(), tol.CHAOTIC_T, tol.CHAOTIC_REFERENCE_TOL)


def regular_agreement_errors() -> np.ndarray:
    """The regular-state agreement measurement of D-046(b)."""
    return agreement_errors(regular_states(), tol.REGULAR_AGREE_T, tol.CONV_REFERENCE_TOL)


@dataclass(frozen=True)
class WiringMeasurement:
    """Wiring-check numbers for one case (D-047)."""

    agreement: float
    """Worst-case ``|RK4 - wrapper|`` on the state ``(theta1, theta2, omega1 t_s, omega2 t_s)``."""
    energy_rk4: float
    """``max_t |E(t) - E(0)| / E_s`` along the RK4 trajectory."""
    energy_wrapper: float
    """The same along the wrapper's trajectory."""


def _energy_drift(rows: np.ndarray, case: WiringCase) -> float:
    """Worst ``|E(t) - E(0)| / E_s`` along ``rows``, with ``E`` under the case's TRUE parameters."""
    energies = np.array([float(_energy(jnp.asarray(row), case.params)) for row in rows])
    return float(np.max(np.abs(energies - energies[0])) / energy_scale(case.params))


def measure_wiring(
    case: WiringCase,
    *,
    rk4_params: dp.Params | None = None,
    wrapper_params: dp.Params | None = None,
) -> WiringMeasurement:
    """Measure the wiring check for one case.

    ``rk4_params`` and ``wrapper_params`` are the parameters each integrator is *given*; they
    default to the case's true parameters. The step, the output times and the energy always use
    the true ones, so the negative controls can hand an integrator the wrong parameters and see
    what the checks make of it.
    """
    rk4_given = case.params if rk4_params is None else rk4_params
    wrapper_given = case.params if wrapper_params is None else wrapper_params
    n_out = round(tol.WIRE_T)
    if n_out != tol.WIRE_T:
        raise ValueError(f"WIRE_T = {tol.WIRE_T!r} must be a whole number of t_s")
    rk4 = np.asarray(
        integrators.rk4_trajectory(
            dp.vector_field,
            jnp.asarray(case.state),
            rk4_given,
            tol.WIRE_DT * case.t_s,
            n_out,
            steps_per_unit(tol.WIRE_DT),
        )
    )
    times = case.t_s * np.arange(n_out + 1, dtype=np.float64)
    wrapper = integrators.solve_reference(
        dp.vector_field,
        case.state,
        wrapper_given,
        times,
        rtol=tol.WIRE_REFERENCE_TOL,
        atol=tol.WIRE_REFERENCE_TOL,
    )
    to_nondimensional = np.array([1.0, 1.0, case.t_s, case.t_s])
    return WiringMeasurement(
        max_abs_diff(rk4 * to_nondimensional, wrapper * to_nondimensional),
        _energy_drift(rk4, case),
        _energy_drift(wrapper, case),
    )


def measure_all_wiring(
    *,
    rk4_params: Callable[[dp.Params], dp.Params] | None = None,
    wrapper_params: Callable[[dp.Params], dp.Params] | None = None,
) -> tuple[WiringMeasurement, ...]:
    """Measure every wiring case. The callables map the true parameters to the ones given."""
    return tuple(
        measure_wiring(
            case,
            rk4_params=None if rk4_params is None else rk4_params(case.params),
            wrapper_params=None if wrapper_params is None else wrapper_params(case.params),
        )
        for case in wiring_cases()
    )
