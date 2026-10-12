"""Tests R1 to R5 (RK4) and S1 to S4 (the DOP853 wrapper) of ``horizon.integrators``.

They stay red until the implementation lands (D-048). Each test says in its docstring what it
catches and what it is blind to. The identity checks go through ``check_close`` with the
IDENTITY class, so no test can quietly use a looser rule. The behaviour over a horizon is
checked by the gates in ``tests/test_integrator_gates.py``; this module checks one step, the
indexing and the contract, and the wrapper's handling of its arguments.

R1 and R2 compare a single step with a formula derived in ``docs/derivations/integrators.md``
(section 2), so they hold whatever any integrator does. R3 compares the trajectory with
repeated steps. S1 to S4 each need SciPy's behaviour on a real input: a blow-up for S3 and a
too-small ``rtol`` for S4.
"""

import inspect
from collections.abc import Callable, Sequence
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from horizon import integrators
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from horizon.tolerances import check_close
from tests import cases
from tests import integrator_checks as chk


def _identity(x: object, x_ref: object, what: str) -> None:
    """Compare against an exact identity: class IDENTITY."""
    check_close(
        np.asarray(x, dtype=np.float64),
        x_ref,
        rtol=tol.IDENTITY_RTOL,
        atol=tol.IDENTITY_ATOL,
        what=what,
    )


# A non-symmetric 3 x 3 matrix with entries of order one, so that every power of ``h A`` in the
# stability polynomial contributes at the step sizes used.
LINEAR_3 = np.array([[-0.7, 1.9, 0.3], [-1.1, 0.4, 2.2], [0.5, -1.3, -0.2]])
STATE_3 = np.array([0.8, -0.4, 1.3])

# A fixed state and parameters with all five parameters different, away from the frozen cases.
PENDULUM_STATE = np.array([2.0, -1.0, 0.3, 0.1])
PENDULUM_PARAMS = dp.Params(1.2, 0.7, 0.9, 1.4, 1.1)


def _f_linear3(y: jax.Array, params: None) -> jax.Array:
    return jnp.asarray(LINEAR_3) @ y


def _f_polynomial(y: jax.Array, params: None) -> jax.Array:
    """A cubic field with rational coefficients: ``(y1 - y0^2 y1 + 1/2, -y0 + y1^3/3 - y0 y1)``."""
    return jnp.stack([y[1] - y[0] ** 2 * y[1] + 0.5, -y[0] + y[1] ** 3 / 3.0 - y[0] * y[1]])


def _f_polynomial_exact(y: Sequence[Fraction]) -> list[Fraction]:
    return [y[1] - y[0] ** 2 * y[1] + Fraction(1, 2), -y[0] + y[1] ** 3 / 3 - y[0] * y[1]]


def _taylor4(z: np.ndarray) -> np.ndarray:
    """``R(z) = I + z + z^2/2 + z^3/6 + z^4/24`` for a square matrix ``z`` (section 2.2)."""
    z2 = z @ z
    return np.eye(len(z)) + z + z2 / 2.0 + z2 @ z / 6.0 + z2 @ z2 / 24.0


def _rk4_step_exact(
    f: Callable[[Sequence[Fraction]], list[Fraction]], y: Sequence[Fraction], h: Fraction
) -> list[Fraction]:
    """The scheme of section 2.1 in exact rational arithmetic, written independently."""

    def shifted(base: Sequence[Fraction], k: Sequence[Fraction], c: Fraction) -> list[Fraction]:
        return [b + c * ki for b, ki in zip(base, k, strict=True)]

    k1 = f(y)
    k2 = f(shifted(y, k1, h / 2))
    k3 = f(shifted(y, k2, h / 2))
    k4 = f(shifted(y, k3, h))
    return [
        yi + h / 6 * (a + 2 * b + 2 * c + d)
        for yi, a, b, c, d in zip(y, k1, k2, k3, k4, strict=True)
    ]


# ---- R1 to R2: one step -------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.3, 0.05])
def test_r1_one_step_on_a_linear_problem_is_the_degree_four_taylor_polynomial(dt: float) -> None:
    """R1 (class IDENTITY): a step on ``y' = A y`` equals ``R(h A) y``, ``R`` the Taylor
    polynomial of ``exp`` through degree 4 (section 2.2).

    Catches a wrong weight, a stage evaluated at the wrong point, and a missing stage, whenever
    the error changes the polynomial. Blind to a wrong tableau that has the same polynomial
    (section 2.4): the nonlinear checks R2 and G1 exist for that.
    """
    step = integrators.rk4_step(_f_linear3, jnp.asarray(STATE_3), None, dt)
    _identity(step, _taylor4(dt * LINEAR_3) @ STATE_3, f"one step, dt = {dt}")


def test_r2_one_step_on_a_nonlinear_field_equals_exact_rational_arithmetic() -> None:
    """R2 (class IDENTITY): a step on a cubic field equals the same scheme in exact arithmetic.

    The field has rational coefficients and the inputs are exact binary numbers, so the reference
    step is computed with ``fractions.Fraction`` and carries no rounding at all. ``h = 0.5`` is
    large on purpose: schemes that differ in a stage differ visibly in one step.

    Catches any slip in the stage points or the weights on a nonlinear problem, including the
    wrong tableau of section 2.4. Blind to a mistake in the scheme as written in section 2.1,
    because it implements the same scheme; the order measured by G1 checks the scheme itself.
    """
    state = np.array([0.5, -0.3])
    h = 0.5
    expected = _rk4_step_exact(
        _f_polynomial_exact, [Fraction(float(v)) for v in state], Fraction(h)
    )
    step = integrators.rk4_step(_f_polynomial, jnp.asarray(state), None, h)
    _identity(step, [float(v) for v in expected], "one step on the cubic field")


# ---- R3: the trajectory ---------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.013, -0.013], ids=["forward", "backward"])
@pytest.mark.parametrize("n_out, stride", [(5, 7), (1, 1), (3, 1), (1, 4)])
def test_r3_rows_are_the_state_after_k_times_stride_steps(
    n_out: int, stride: int, dt: float
) -> None:
    """R3 (class IDENTITY): row ``k`` is the state after ``k * stride`` calls of ``rk4_step``.

    Catches an off-by-one in the number of steps, a wrong stride, a missing or duplicated initial
    row, and a trajectory that drifts from the step it is built on. Row 0 must be the initial
    state exactly. Blind to an error that both functions share.
    """
    step = jax.jit(integrators.rk4_step, static_argnums=0)
    state0 = jnp.asarray(PENDULUM_STATE)
    rows = np.asarray(
        integrators.rk4_trajectory(dp.vector_field, state0, PENDULUM_PARAMS, dt, n_out, stride)
    )
    assert rows.shape == (n_out + 1, 4)
    assert np.array_equal(rows[0], PENDULUM_STATE), "row 0 must be the initial state, exactly"
    state = state0
    for k in range(1, n_out + 1):
        for _ in range(stride):
            state = step(dp.vector_field, state, PENDULUM_PARAMS, dt)
        _identity(rows[k], np.asarray(state), f"row {k} ({k * stride} steps)")


# ---- R4: shapes, dtype, jit and vmap --------------------------------------------------------


def test_r4_shapes_and_dtype() -> None:
    """R4 (class IDENTITY): documented shapes and ``float64``, for a 2-vector and a 4-vector.

    Catches a transposed or flattened result, a missing initial row, and a silent ``float32``.
    Blind to values.
    """
    for f, state, params in (
        (chk.f_harmonic, np.array([1.0, 0.5]), None),
        (dp.vector_field, PENDULUM_STATE, PENDULUM_PARAMS),
    ):
        n = len(state)
        step = integrators.rk4_step(f, jnp.asarray(state), params, 0.01)
        rows = integrators.rk4_trajectory(f, jnp.asarray(state), params, 0.01, 3, 4)
        assert step.shape == (n,)
        assert rows.shape == (4, n)
        assert step.dtype == jnp.float64
        assert rows.dtype == jnp.float64


def test_r4_jit_agrees_with_eager_calls() -> None:
    """R4 (class IDENTITY): wrapping both functions in ``jax.jit`` changes nothing.

    Catches Python control flow on a traced value, and a function that only works eagerly. Blind
    to behaviour that depends on the first compilation.
    """
    state = jnp.asarray(PENDULUM_STATE)
    jit_step = jax.jit(integrators.rk4_step, static_argnames=("f",))
    jit_rows = jax.jit(integrators.rk4_trajectory, static_argnames=("f", "n_out", "stride"))
    _identity(
        jit_step(dp.vector_field, state, PENDULUM_PARAMS, 0.02),
        integrators.rk4_step(dp.vector_field, state, PENDULUM_PARAMS, 0.02),
        "jit step",
    )
    _identity(
        jit_rows(dp.vector_field, state, PENDULUM_PARAMS, 0.02, 3, 4),
        integrators.rk4_trajectory(dp.vector_field, state, PENDULUM_PARAMS, 0.02, 3, 4),
        "jit trajectory",
    )


def test_r4_vmap_over_states_parameters_and_dt_matches_single_calls() -> None:
    """R4 (class IDENTITY): ``jax.vmap`` over states, over a ``Params`` of arrays, and over ``dt``.

    Catches a ``params`` or ``dt`` that was made static, closed over, or silently shared between
    batch members. Vmapping over ``dt`` works only if ``dt`` is traced. Blind to compile cost.
    """
    subset = cases.moderate_cases()[:3]
    states = jnp.asarray(np.stack([case.states[0] for case in subset]))
    batched = dp.Params(*(jnp.array([case.params[i] for case in subset]) for i in range(5)))
    dts = jnp.array([0.01, 0.02, 0.03])

    over_states = jax.vmap(
        lambda y: integrators.rk4_trajectory(dp.vector_field, y, dp.DEFAULT_PARAMS, 0.01, 3, 5)
    )(states)
    over_params = jax.vmap(
        lambda y, p: integrators.rk4_trajectory(dp.vector_field, y, p, 0.01, 3, 5)
    )(states, batched)
    over_dt = jax.vmap(
        lambda y, h: integrators.rk4_trajectory(dp.vector_field, y, dp.DEFAULT_PARAMS, h, 3, 5)
    )(states, dts)

    for i, case in enumerate(subset):
        state = jnp.asarray(case.states[0])
        params = dp.Params(*case.params)
        _identity(
            over_states[i],
            integrators.rk4_trajectory(dp.vector_field, state, dp.DEFAULT_PARAMS, 0.01, 3, 5),
            f"vmap over states, member {i}",
        )
        _identity(
            over_params[i],
            integrators.rk4_trajectory(dp.vector_field, state, params, 0.01, 3, 5),
            f"vmap over parameters, member {i}",
        )
        _identity(
            over_dt[i],
            integrators.rk4_trajectory(
                dp.vector_field, state, dp.DEFAULT_PARAMS, float(dts[i]), 3, 5
            ),
            f"vmap over dt, member {i}",
        )


# ---- R5: argument checks and negative steps -------------------------------------------------


@pytest.mark.parametrize("n_out, stride", [(0, 1), (1, 0), (-1, 2), (2, -3)])
def test_r5_counts_below_one_are_rejected(n_out: int, stride: int) -> None:
    """R5 (class exact): ``n_out`` and ``stride`` below 1 raise ``ValueError``.

    Catches the silent case: zero steps per output would return the initial state in every row
    with no warning. Blind to non-integer counts (next test).
    """
    with pytest.raises(ValueError):
        integrators.rk4_trajectory(chk.f_harmonic, jnp.array([1.0, 0.5]), None, 0.1, n_out, stride)


@pytest.mark.parametrize("n_out, stride", [(2.0, 1), (1, 2.5)])
def test_r5_counts_that_are_not_integers_are_rejected(n_out: float, stride: float) -> None:
    """R5 (class exact): a ``float`` count raises ``TypeError``.

    Catches a count that reaches ``lax.scan`` as a float, which fails with a message about
    ``scan`` and not about the argument. Blind to numpy integer types, which the interface leaves
    undefined ("Python ints", D-042).
    """
    with pytest.raises(TypeError):
        integrators.rk4_trajectory(chk.f_harmonic, jnp.array([1.0, 0.5]), None, 0.1, n_out, stride)


def test_r5_a_negative_dt_is_legal_and_follows_the_same_formula() -> None:
    """R5 (class IDENTITY): ``dt < 0`` gives ``R(-|h| A) y``; there is no sign check on ``dt``.

    Milestone 4's reversibility tests integrate backward. Catches a ``dt`` validated or taken
    through ``abs``. Blind to accuracy of backward integration, which is the same scheme.
    """
    h = 0.2
    step = integrators.rk4_step(_f_linear3, jnp.asarray(STATE_3), None, -h)
    _identity(step, _taylor4(-h * LINEAR_3) @ STATE_3, "one backward step")


# ---- S1 to S4: the DOP853 wrapper -----------------------------------------------------------


def _decay(y: jax.Array, rate: float) -> jax.Array:
    """``y' = -rate y``: ``params`` is a plain number, so that S1 can see it arrive."""
    return -rate * y


def _f_blow_up(y: jax.Array, params: None) -> jax.Array:
    """``y' = y^2``, ``y(0) = 1``: the solution ``1 / (1 - t)`` is infinite at ``t = 1``."""
    return y**2


def test_s1_contract_signature_defaults_shape_dtype_and_params() -> None:
    """S1 (class IDENTITY): documented signature, shape, ``float64`` NumPy result, params arrive.

    ``rtol`` and ``atol`` are keyword-only with the default ``1e-12`` (brief section 3). The
    solution of ``y' = -3 y`` from ``(2, -1)`` is compared with ``exp(-3 t)``, so a ``params``
    that is dropped or not passed on to ``f`` fails. Row 0 is the initial state.

    Catches a wrong default, a positional tolerance, a transposed result, and a ``jax`` array
    returned instead of NumPy. Blind to the accuracy at other tolerances (G1).
    """
    parameters = inspect.signature(integrators.solve_reference).parameters
    for name in ("rtol", "atol"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default == 1e-12

    state0 = np.array([2.0, -1.0])
    times = np.array([0.0, 0.5, 1.0])
    rows = integrators.solve_reference(_decay, state0, 3.0, times)
    assert isinstance(rows, np.ndarray)
    assert rows.dtype == np.float64
    assert rows.shape == (3, 2)
    _identity(rows[0], state0, "row 0 is the initial state")
    chk.check_at_most(
        chk.max_abs_diff(rows, np.exp(-3.0 * times)[:, None] * state0),
        tol.EXACT_WRAPPER_TOL,
        what="wrapper against exp(-3 t)",
    )


@pytest.mark.parametrize(
    "times",
    [[], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0], [-0.5, 1.0]],
    ids=["empty", "repeated", "decreasing", "negative-first"],
)
def test_s2_a_t_eval_that_is_empty_not_increasing_or_starts_below_zero_is_rejected(
    times: list[float],
) -> None:
    """S2 (class exact): ``ValueError`` unless ``t_eval`` is non-empty, ``t_eval[0] >= 0`` and
    strictly increasing.

    Catches a ``t_eval`` that SciPy would handle in a surprising way (an empty one fails on
    ``t_eval[-1]``, a decreasing one integrates backward). Blind to a 2-D ``t_eval``.
    """
    with pytest.raises(ValueError):
        integrators.solve_reference(_decay, np.array([1.0]), 1.0, np.array(times))


def test_s2_a_first_time_above_zero_is_accepted() -> None:
    """S2 (class IDENTITY): ``t_eval = (0.5, 1, 2)`` works and is accurate.

    The integration still starts at ``t = 0``, so the first row is the state at 0.5, not the
    initial state. Catches a wrapper that assumes ``t_eval[0] == 0``. Blind to the other rules.
    """
    times = np.array([0.5, 1.0, 2.0])
    rows = integrators.solve_reference(_decay, np.array([2.0]), 3.0, times)
    assert rows.shape == (3, 1)
    chk.check_at_most(
        chk.max_abs_diff(rows, 2.0 * np.exp(-3.0 * times)[:, None]),
        tol.EXACT_WRAPPER_TOL,
        what="wrapper against 2 exp(-3 t) at t = 0.5, 1, 2",
    )


def test_s3_a_failed_solve_raises_a_runtime_error() -> None:
    """S3 (class exact): SciPy reports failure (non-zero status) and the wrapper raises.

    ``y' = y^2`` with ``y(0) = 1`` blows up at ``t = 1``, so integrating to ``t = 2`` makes the
    step size underflow. Catches a wrapper that returns the partial solution, whose rows are
    missing, or fills them. Blind to failures that SciPy reports through a warning instead.

    ``NotImplementedError`` is a subclass of ``RuntimeError``, so an unfinished stub would pass
    ``pytest.raises(RuntimeError)``. The exact type is checked for that reason.
    """
    with pytest.raises(RuntimeError) as raised:
        integrators.solve_reference(_f_blow_up, np.array([1.0]), None, np.array([0.0, 2.0]))
    assert type(raised.value) is RuntimeError


def test_s4_an_rtol_below_scipys_floor_raises_instead_of_being_clamped() -> None:
    """S4 (class exact): ``rtol = 1e-14`` raises SciPy's warning as an exception.

    SciPy raises any ``rtol`` below ``100 * eps = 2.22e-14`` to that value and only warns, so a
    reference run at the wrong tolerance would look fine. Catches a wrapper that lets the warning
    through. Blind to warnings that SciPy does not emit.
    """
    with pytest.raises(UserWarning, match="rtol"):
        integrators.solve_reference(_decay, np.array([1.0]), 1.0, np.array([0.0, 1.0]), rtol=1e-14)


def test_s4_the_self_check_tolerance_is_accepted() -> None:
    """S4 (class exact): ``rtol = atol = CONV_SELF_CHECK_TOL`` runs without a warning.

    D-045 depends on ``2.5e-14`` being just above the floor. Catches a strict mode that is too
    strict. Blind to accuracy at that tolerance, which the validity rule of G2 measures.
    """
    rows = integrators.solve_reference(
        _decay,
        np.array([1.0]),
        1.0,
        np.array([0.0, 1.0]),
        rtol=tol.CONV_SELF_CHECK_TOL,
        atol=tol.CONV_SELF_CHECK_TOL,
    )
    assert rows.shape == (2, 1)
