"""Contracts of ``horizon.lyapunov``, the wrap, and the canonical identities (D-077, D-078).

These are not the gates: the gates and their controls are in ``tests/test_lyapunov_gates.py``.
Here the interface of D-077 is pinned (shapes, dtypes, counts, the window and the seam for the
controls) on cases whose answer is exact, and the canonical coordinates of D-078 are checked
against independent paths at the frozen parameter sets of D-031. Every milestone 5 test is in the
full tier (D-084).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from horizon import integrators, lyapunov
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from horizon.tolerances import check_close
from tests import cases
from tests import lyapunov_checks as lc

MODERATE = cases.moderate_cases()
LARGE_ANGLE = cases.large_angle_cases()
DIAG = jnp.diag(jnp.array([0.3, -0.1, -0.7]))
"""A diagonal linear field: with ``frame0 = I`` the QR factor is diagonal at every interval."""


def _identity(x: object, x_ref: object, *, scale: object, what: str) -> float:
    return check_close(
        np.asarray(x),
        np.asarray(x_ref),
        rtol=tol.IDENTITY_RTOL,
        atol=tol.IDENTITY_ATOL,
        scale=scale,
        what=what,
    )


def _qr(**overrides: object) -> lyapunov.QRResult:
    kwargs: dict[str, object] = dict(dt=0.01, steps_per_qr=10, n_transient=2, n_qr=5)
    kwargs.update(overrides)
    return lyapunov.tangent_qr(lc.linear, jnp.ones(3), DIAG, jnp.eye(3), **kwargs)


def _benettin(**overrides: object) -> lyapunov.BenettinResult:
    kwargs: dict[str, object] = dict(
        dt=0.01,
        d0=1e-9,
        steps_per_renorm=10,
        n_transient=2,
        n_renorm=5,
        angle_mask=lc.NO_ANGLES_3,
    )
    kwargs.update(overrides)
    return lyapunov.benettin(lc.linear, jnp.ones(3), DIAG, jnp.array([1.0, 0.0, 0.0]), **kwargs)


# ---- the wrap (D-010, D-077) ---------------------------------------------------------------


def test_wrap_maps_into_minus_pi_to_pi_and_keeps_small_differences() -> None:
    """``wrap`` lands in ``[-pi, pi)``, is the identity there, and ignores multiples of 2 pi.

    Catches a wrap into ``[0, 2 pi)``, a missing shift, or a wrap applied twice. Blind to the
    choice at exactly ``pi``, which no gate reaches.
    """
    small = jnp.linspace(-3.0, 3.0, 13)
    _identity(lyapunov.wrap(small), small, scale=1.0, what="wrap of small differences")
    for k in (-3, -1, 1, 7):
        shifted = lyapunov.wrap(small + 2.0 * math.pi * k)
        _identity(shifted, small, scale=abs(2.0 * math.pi * k) + 3.0, what=f"wrap, k = {k}")
    wide = lyapunov.wrap(jnp.linspace(-100.0, 100.0, 1001))
    assert wide.dtype == jnp.float64
    assert float(jnp.min(wide)) >= -math.pi and float(jnp.max(wide)) < math.pi


def test_state_difference_wraps_only_the_masked_components() -> None:
    """``state_difference`` wraps where the mask is true and nowhere else (D-077).

    Catches a wrap of the velocities, which would fold a large ``Delta p`` into ``[-pi, pi)``,
    or a wrap of nothing. Blind to the mask's content, which D-078 fixes.
    """
    a = jnp.array([10.0, -10.0, 10.0, -10.0])
    b = jnp.zeros(4)
    got = lyapunov.state_difference(a, b, lc.PENDULUM_MASK)
    want = jnp.array([float(lyapunov.wrap(10.0)), float(lyapunov.wrap(-10.0)), 10.0, -10.0])
    _identity(got, want, scale=10.0, what="state difference")


# ---- the estimators: interface and exact cases (D-077) -------------------------------------


def test_tangent_qr_returns_the_declared_shapes_and_dtypes() -> None:
    """``QRResult`` has the shapes of D-077, in ``float64``; ``log_det_steps`` only on request.

    Catches a transient that is recorded, a missing initial row of ``states`` and a wrong axis.
    """
    plain = _qr()
    assert plain.log_r.shape == (5, 3) and plain.log_r.dtype == jnp.float64
    assert plain.states.shape == (6, 3) and plain.final_state.shape == (3,)
    assert plain.max_abs_state.shape == () and plain.cond_r.shape == (5,)
    assert plain.log_det_steps is None
    with_logdet = _qr(step_logdet=True)
    assert with_logdet.log_det_steps is not None and with_logdet.log_det_steps.shape == (5,)
    two = lyapunov.tangent_qr(
        lc.linear, jnp.ones(3), DIAG, jnp.eye(3)[:, :2], dt=0.01, steps_per_qr=10,
        n_transient=0, n_qr=4,
    )  # fmt: skip
    assert two.log_r.shape == (4, 2)


def test_benettin_returns_the_declared_shapes_and_dtypes() -> None:
    """``BenettinResult`` has the shapes of D-077, in ``float64``."""
    result = _benettin()
    assert result.log_growth.shape == (5,) and result.log_growth.dtype == jnp.float64
    assert result.final_state.shape == (3,) and result.max_abs_state.shape == ()


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(dict(steps_per_qr=0), id="steps-zero"),
        pytest.param(dict(n_qr=0), id="n-qr-zero"),
        pytest.param(dict(n_transient=-1), id="transient-negative"),
        pytest.param(dict(steps_per_qr=2.0), id="steps-float"),
        pytest.param(dict(n_qr=True), id="n-qr-bool"),
    ],
)
def test_tangent_qr_rejects_bad_counts(overrides: dict[str, object]) -> None:
    """Counts are Python ints, at least 1, and ``n_transient`` at least 0 (D-077)."""
    with pytest.raises((TypeError, ValueError)):
        _qr(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(dict(steps_per_renorm=0), id="steps-zero"),
        pytest.param(dict(n_renorm=0), id="n-renorm-zero"),
        pytest.param(dict(n_transient=-1), id="transient-negative"),
        pytest.param(dict(n_renorm=3.0), id="n-renorm-float"),
        pytest.param(dict(angle_mask=(False, False)), id="mask-length"),
    ],
)
def test_benettin_rejects_bad_counts_and_masks(overrides: dict[str, object]) -> None:
    """Counts are Python ints, and the mask has one entry per component (D-077)."""
    with pytest.raises((TypeError, ValueError)):
        _benettin(**overrides)


def test_tangent_qr_rejects_step_logdet_with_a_partial_frame() -> None:
    """``step_logdet=True`` needs ``k = n`` (D-077)."""
    with pytest.raises(ValueError):
        lyapunov.tangent_qr(
            lc.linear, jnp.ones(3), DIAG, jnp.eye(3)[:, :2], dt=0.01, steps_per_qr=10,
            n_transient=0, n_qr=2, step_logdet=True,
        )  # fmt: skip


def test_tangent_qr_on_a_diagonal_field_gives_the_rk4_map_exactly() -> None:
    """With ``A`` diagonal and ``frame0 = I``, ``ln|R_ii| = N_Q ln|R(dt mu_i)|`` per interval.

    The exact answer of the RK4 map (note, section 1.3). Catches a tangent step that is not the
    derivative of ``rk4_step``, a wrong interval length, and a transient that leaks into the
    record. Blind to anything a diagonal ``A`` cannot show (the QR rotation).
    """
    result = _qr(step_logdet=True)
    mu = np.diag(np.asarray(DIAG))
    per_interval = 10 * np.log(np.abs(lc.rk4_stability(0.01 * mu)))
    want = np.broadcast_to(per_interval, (5, 3))
    _identity(result.log_r, want, scale=np.abs(want), what="ln|R_ii| on a diagonal field")
    _identity(
        result.log_det_steps,
        np.full(5, per_interval.sum()),
        scale=np.abs(per_interval).sum(),
        what="log_det_steps on a diagonal field",
    )
    _identity(result.cond_r, np.full(5, np.exp(np.ptp(per_interval))), scale=10.0, what="cond")


def test_tangent_qr_states_follow_rk4_trajectory_from_the_end_of_the_transient() -> None:
    """``states`` are the RK4 states at the QR times after the transient (D-077).

    Catches an off-by-one interval, a state advanced by the tangent map, and a ``max_abs_state``
    taken over the transient.
    """
    result = _qr()
    rows = integrators.rk4_trajectory(lc.linear, jnp.ones(3), DIAG, 0.01, 7, 10)
    _identity(result.states, rows[2:], scale=1.0, what="QR states")
    _identity(result.final_state, rows[-1], scale=1.0, what="QR final state")
    fine = integrators.rk4_trajectory(lc.linear, jnp.ones(3), DIAG, 0.01, 70, 1)[20:]
    _identity(result.max_abs_state, jnp.max(jnp.abs(fine)), scale=1.0, what="max_abs_state")


def test_benettin_on_a_diagonal_field_gives_the_rk4_map_exactly() -> None:
    """Along an eigenvector, ``ln(d / d0) = N_B ln|R(dt mu_1)|`` up to rounding of the difference.

    Catches a missing renormalization, a log of the squared distance and a transient recorded.
    The tolerance is the worst-case rounding of D-085 per interval, ``s eps X N_B / d0``.
    """
    result = _benettin()
    want = np.full(5, 10 * math.log(abs(lc.rk4_stability(0.01 * 0.3))))
    x = float(result.max_abs_state)
    check_close(
        np.asarray(result.log_growth),
        want,
        rtol=0.0,
        atol=tol.DELTA_BQ_S * lc.EPS * x * 10 / 1e-9,
        what="Benettin on a diagonal field",
    )
    rows = integrators.rk4_trajectory(lc.linear, jnp.ones(3), DIAG, 0.01, 7, 10)
    _identity(result.final_state, rows[-1], scale=1.0, what="Benettin final state")


def test_exponents_and_running_exponents_average_over_the_recorded_intervals() -> None:
    """``exponents`` is the mean over the interval; ``running_exponents`` its cumulative form."""
    terms = jnp.array([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]])
    _identity(lyapunov.exponents(terms, 2.0), [1.5, -2.0], scale=6.0, what="exponents")
    want = np.array([[0.5, -1.0], [1.0, -1.5], [1.5, -2.0]])
    _identity(lyapunov.running_exponents(terms, 2.0), want, scale=6.0, what="running")


def test_the_private_core_accepts_another_tangent_step() -> None:
    """The seam of D-077: a tangent step injected into the core changes ``log_r`` only.

    The state and ``log_det_steps`` keep following RK4, so the Euler-frame control of D-079 goes
    through the same measurement.
    """
    euler = lc.qr_with(
        lc.linear, jnp.ones(3), DIAG, jnp.eye(3), dt=0.01, steps_per_qr=10, n_transient=2,
        n_qr=5, tangent_step=lc.euler_tangent_step,
    )  # fmt: skip
    plain = _qr(step_logdet=True)
    _identity(euler.states, plain.states, scale=1.0, what="states under the Euler frame")
    _identity(euler.log_det_steps, plain.log_det_steps, scale=1.0, what="log_det_steps")
    assert float(jnp.max(jnp.abs(euler.log_r - plain.log_r))) > 1e-6


# ---- canonical coordinates (D-078) ---------------------------------------------------------


def _ids(case: cases.Case) -> str:
    return f"set-{case.index:02d}"


def _canonical_checks(
    params_tuple: tuple[float, ...], states: np.ndarray, rtol: float, atol: float
) -> None:
    params = dp.Params(*params_tuple)

    def close(x: object, ref: object, scale: object, what: str) -> None:
        check_close(np.asarray(x), np.asarray(ref), rtol=rtol, atol=atol, scale=scale, what=what)

    j = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    for i, x in enumerate(jnp.asarray(states)):
        z = dp.to_canonical(x, params)
        back = dp.from_canonical(z, params)
        close(back, x, np.abs(np.asarray(x)) + 1.0, f"round trip, state {i}")

        jac = np.asarray(jax.jacfwd(dp.to_canonical)(x, params))
        f = np.asarray(dp.vector_field(x, params))
        lhs = np.asarray(dp.canonical_vector_field(z, params))
        close(lhs, jac @ f, np.abs(jac) @ np.abs(f), f"conjugacy, state {i}")

        grad_h = np.asarray(jax.grad(dp.hamiltonian)(z, params))
        close(lhs, j @ grad_h, np.abs(lhs) + np.abs(j) @ np.abs(grad_h), f"Hamilton, state {i}")

        h = float(dp.hamiltonian(z, params))
        e = float(dp.energy(x, params))
        scale = float(dp.kinetic_energy(x, params)) + abs(float(dp.potential_energy(x, params)))
        close(h, e, scale, f"H = E, state {i}")

        dfz = np.asarray(jax.jacfwd(dp.canonical_vector_field)(z, params))
        close(np.trace(dfz), 0.0, np.sum(np.abs(np.diag(dfz))), f"zero trace, state {i}")


@pytest.mark.parametrize("case", MODERATE, ids=_ids)
def test_canonical_coordinates_satisfy_their_identities(case: cases.Case) -> None:
    """D-078 (class IDENTITY): round trip, conjugacy, Hamilton's equations, ``H = E``, zero trace.

    Each at every state of the set; the scale is the sum of the absolute values of the cancelled
    terms (D-031). Hamilton's equations use ``jax.grad`` of ``hamiltonian``, a path independent
    of ``canonical_vector_field``. Catches a wrong ``M'``, a transposed ``M`` and a sign in
    ``p'``. Blind to an error shared by ``acceleration`` and ``energy``, which milestone 2 tests.
    """
    _canonical_checks(case.params, case.states, tol.IDENTITY_RTOL, tol.IDENTITY_ATOL)


@pytest.mark.parametrize("case", LARGE_ANGLE, ids=lambda c: f"large-angle-{c.index}")
def test_canonical_coordinates_satisfy_their_identities_at_large_unwrapped_angles(
    case: cases.Case,
) -> None:
    """D-078 (class IMPL): the same identities at the large-angle states of D-032."""
    _canonical_checks(case.params, case.states, tol.IMPL_RTOL, tol.IMPL_ATOL)
