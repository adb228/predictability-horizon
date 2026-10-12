"""Gates G1 to G6, closed-form checks X1 to X4 and negative controls N1 to N3 (D-044 to D-047).

The thresholds, grids, horizons and seeds were declared in the decision log before any
implementation existed and live in ``horizon.tolerances`` (section 7 of
``docs/derivations/integrators.md``). A gate that fails is reported, never adjusted (D-025).

Which tests need the implementation: G1 to G6 and N2, N3 do. X1 to X4 and N1 do not: X checks
each closed-form solution against its own equation with ``jax.jvp``, and N1 runs deliberately
wrong schemes written here through the same gate machinery, to show that the gate can fail. They
pass on an interface skeleton, which is what they should do.

The measurements and the checks are in ``tests/integrator_checks.py``, shared with the script
``experiments/stage1/integrator_validation.py``, so a gate and the number printed for it are the
same code.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from horizon.tolerances import check_close
from tests import integrator_checks as chk

PROBLEMS = chk.exact_problems()


def _problem_id(problem: chk.ExactProblem) -> str:
    return problem.name


# ---- X1 to X4: the closed forms satisfy their equations -------------------------------------


@pytest.mark.parametrize("problem", PROBLEMS, ids=_problem_id)
def test_x_closed_form_satisfies_its_equation_and_initial_condition(
    problem: chk.ExactProblem,
) -> None:
    """X1 to X4 (class IDENTITY): ``d/dt exact(t) = f(exact(t))`` and ``exact(0) = y0``.

    The derivative is taken with ``jax.jvp``, so no integrator is involved. Catches a typo in a
    closed form, which would otherwise make the gate compare RK4 with a wrong "exact" solution.
    Blind to a closed form that satisfies the equation but belongs to another initial condition
    (the second assertion covers that).
    """
    kwargs = {"rtol": tol.IDENTITY_RTOL, "atol": tol.IDENTITY_ATOL}
    for t in (0.0, 0.3, 1.0, 2.7, float(problem.horizon)):
        time = jnp.float64(t)
        value, derivative = jax.jvp(problem.exact, (time,), (jnp.ones_like(time),))
        check_close(
            np.asarray(derivative),
            np.asarray(problem.f(value, None)),
            what=f"{problem.name}: d/dt exact versus f(exact) at t = {t}",
            **kwargs,
        )
    check_close(
        np.asarray(problem.exact(jnp.float64(0.0))),
        problem.y0,
        what=f"{problem.name}: exact(0) versus y0",
        **kwargs,
    )


# ---- G1: RK4 against exact solutions (D-044) ------------------------------------------------


@pytest.mark.parametrize("problem", PROBLEMS, ids=_problem_id)
def test_g1_rk4_converges_at_order_four_on_the_exact_problems(problem: chk.ExactProblem) -> None:
    """G1 (class slope band and ceiling): RK4's error has slope 4 and stays under its ceiling.

    The slope is the least-squares slope of ``log(error)`` against ``log(dt)`` over the grid of
    D-044, and must be in ``EXACT_SLOPE_BAND``; the error at the finest step must be at most the
    problem's ceiling. No reference solver and no pendulum is involved.

    Catches a wrong order, a wrong constant factor, and a scheme that is only accurate on linear
    problems (the two nonlinear problems). Blind to defects below the ceilings.
    """
    chk.check_exact_rk4(problem, chk.rk4_exact_errors(problem))


@pytest.mark.parametrize("problem", PROBLEMS, ids=_problem_id)
def test_g1_wrapper_is_within_the_declared_distance_of_the_exact_solution(
    problem: chk.ExactProblem,
) -> None:
    """G1 (class ceiling): the DOP853 wrapper at its default tolerances is near the exact solution.

    This is the evidence that the reference can be trusted at the brief's tolerances on problems
    where the truth is known. Catches a wrapper that mis-handles ``t_eval`` or ``params``, and a
    wrong default tolerance. Blind to chaotic motion (G3).
    """
    chk.check_at_most(
        chk.wrapper_exact_error(problem),
        tol.EXACT_WRAPPER_TOL,
        what=f"{problem.name}: wrapper against the exact solution",
    )


# ---- G2: convergence on the pendulum (D-045) ------------------------------------------------


def test_g2_rk4_converges_at_order_four_on_regular_pendulum_states() -> None:
    """G2 (class slope band and validity rule): every regular state's slope is in the band.

    Per state the slope of ``log(error)`` against ``log(dt)`` is measured against a ``1e-13``
    reference. The validity rule comes first: the error at the finest step must be at least
    ``CONV_FLOOR_RATIO`` times the reference's own error (the distance between the ``1e-13`` and
    ``2.5e-14`` references), otherwise the measurement is at the reference's noise floor and the
    test fails as inconclusive instead of passing. All violations are reported together.

    Catches an order defect on the actual pendulum field and a reference too noisy to judge.
    Blind to nonlinear defects that small amplitudes cannot excite: G1 carries those.
    """
    result = chk.measure_convergence()
    finest = int(np.argmin(tol.CONV_DT_GRID))
    failures = []
    for i in range(len(result.slopes)):
        checks = (
            partial(
                chk.check_at_least,
                result.errors[i, finest],
                tol.CONV_FLOOR_RATIO * result.self_errors[i],
                what=f"state {i}: INCONCLUSIVE, error at dt = {tol.CONV_DT_GRID[finest]} against"
                f" {tol.CONV_FLOOR_RATIO:g} times the reference's own error",
            ),
            partial(
                chk.check_in_band,
                result.slopes[i],
                tol.CONV_SLOPE_BAND,
                what=f"state {i}: convergence slope",
            ),
        )
        for check in checks:
            try:
                check()
            except AssertionError as error:
                failures.append(str(error))
    assert not failures, "\n".join(failures)


# ---- G3 and G4: agreement of RK4 and DOP853 (D-046) -----------------------------------------


def test_g3_rk4_and_dop853_agree_on_chaotic_ish_states_over_the_declared_horizon() -> None:
    """G3 (class ceiling): the worst case over 30 states, 9 output times and 4 components.

    The horizon ``T = 8`` and the tolerance ``1e-4`` were declared together (D-046): chaos makes
    the agreement time depend only on the logarithm of the tolerance, so neither means anything
    alone. Catches an RK4 or reference defect that shows on strongly nonlinear motion. Blind to
    defects below the tolerance, and to swapped parameters (all parameters are 1 here).
    """
    errors = chk.chaotic_agreement_errors()
    chk.check_at_most(
        float(np.max(errors)),
        tol.CHAOTIC_TOL,
        what=f"chaotic-ish states, worst |RK4 - DOP853| up to t = {tol.CHAOTIC_T}",
    )


def test_g4_rk4_and_dop853_agree_on_regular_states_over_the_declared_horizon() -> None:
    """G4 (class ceiling): the worst case over the 12 regular states up to ``t = 50``.

    Regular motion is close to a pair of linear oscillators, so errors grow slowly and a long
    horizon is meaningful. Catches slow phase drift from a defect too small for G3. Blind to
    swapped parameters (all parameters are 1 here).
    """
    errors = chk.regular_agreement_errors()
    chk.check_at_most(
        float(np.max(errors)),
        tol.REGULAR_AGREE_TOL,
        what=f"regular states, worst |RK4 - DOP853| up to t = {tol.REGULAR_AGREE_T}",
    )


# ---- G5 and G6: non-default parameters (D-047) ----------------------------------------------


@pytest.fixture(scope="module")
def wiring() -> tuple[chk.WiringMeasurement, ...]:
    """The wiring measurements of all 30 cases, computed once for G5 and G6."""
    return chk.measure_all_wiring()


def test_g5_rk4_and_the_wrapper_agree_on_non_default_parameters(
    wiring: tuple[chk.WiringMeasurement, ...],
) -> None:
    """G5 (class ceiling): RK4 and the wrapper agree on ``(theta, omega * t_s)`` for 10 sets.

    Catches a dropped, swapped or unused ``params`` in only one of the two integrators. Blind to
    the same error in both (N3): that is what the energy anchor, G6, is for.
    """
    chk.check_at_most(
        max(m.agreement for m in wiring),
        tol.WIRE_AGREE_TOL,
        what="non-default parameters, worst |RK4 - wrapper| in nondimensional state",
    )


def test_g6_rk4_conserves_the_energy_of_its_own_parameters(
    wiring: tuple[chk.WiringMeasurement, ...],
) -> None:
    """G6 (class ceiling): ``max_t |E(t) - E(0)| / E_s`` along the RK4 trajectory.

    ``E`` is ``double_pendulum.energy`` under the set's own parameters and ``E_s`` is the largest
    possible ``|V|``. Catches an integrator that follows the wrong dynamics, whether or not any
    other integrator agrees with it. Blind to an error that conserves energy.
    """
    chk.check_at_most(
        max(m.energy_rk4 for m in wiring),
        tol.WIRE_ENERGY_TOL,
        what="non-default parameters, RK4 energy drift over E_s",
    )


def test_g6_the_wrapper_conserves_the_energy_of_its_own_parameters(
    wiring: tuple[chk.WiringMeasurement, ...],
) -> None:
    """G6 (class ceiling): the same energy anchor along the DOP853 wrapper's trajectory.

    Catches a wrapper that drops or misroutes ``params``. Blind to an error that conserves
    energy.
    """
    chk.check_at_most(
        max(m.energy_wrapper for m in wiring),
        tol.WIRE_ENERGY_TOL,
        what="non-default parameters, wrapper energy drift over E_s",
    )


# ---- N1: deliberately wrong schemes (the exact-solution gate can fail) ----------------------
#
# Written here, independently of ``horizon.integrators``. Each takes the same arguments as
# ``rk4_step``. They are not meant to be good schemes.


def _scheme_w1(f: Callable, y: jax.Array, params: object, dt: float) -> jax.Array:
    """``k3`` is evaluated at ``y + dt/2 k1`` instead of ``y + dt/2 k2``: second order."""
    k1 = f(y, params)
    k2 = f(y + 0.5 * dt * k1, params)
    k3 = f(y + 0.5 * dt * k1, params)
    k4 = f(y + dt * k3, params)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _scheme_w2(f: Callable, y: jax.Array, params: object, dt: float) -> jax.Array:
    """Every stage fraction 1/2 and weights ``(0, 1/3, 1/3, 1/3)`` (section 2.4).

    The same stability polynomial as RK4, so fourth order on linear constant-coefficient
    problems; ``sum b c^2 = 1/4`` instead of ``1/3``, so second order in general.
    """
    k1 = f(y, params)
    k2 = f(y + 0.5 * dt * k1, params)
    k3 = f(y + 0.5 * dt * k2, params)
    k4 = f(y + 0.5 * dt * k3, params)
    return y + dt * (k2 + k3 + k4) / 3.0


def _scheme_w3(f: Callable, y: jax.Array, params: object, dt: float) -> jax.Array:
    """``k4`` is evaluated at ``y + dt/2 k3`` instead of ``y + dt k3``: first order."""
    k1 = f(y, params)
    k2 = f(y + 0.5 * dt * k1, params)
    k3 = f(y + 0.5 * dt * k2, params)
    k4 = f(y + 0.5 * dt * k3, params)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


WRONG_SCHEMES = {
    # name: (step, problems on which the gate is expected to pass)
    "W1-k3-from-k1": (_scheme_w1, frozenset()),
    "W2-same-polynomial": (_scheme_w2, frozenset({"harmonic", "linear"})),
    "W3-k4-half-step": (_scheme_w3, frozenset()),
}


def _trajectory_from_step(step: Callable) -> chk.Trajectory:
    """Build a trajectory function with the signature of ``rk4_trajectory`` from a step."""

    @partial(jax.jit, static_argnames=("f", "n_out", "stride"))
    def trajectory(
        f: Callable, state0: jax.Array, params: object, dt: float, n_out: int, stride: int
    ) -> jax.Array:
        def inner(y: jax.Array, _: None) -> tuple[jax.Array, None]:
            return step(f, y, params, dt), None

        def outer(y: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
            y, _ = jax.lax.scan(inner, y, None, length=stride)
            return y, y

        _, rows = jax.lax.scan(outer, state0, None, length=n_out)
        return jnp.concatenate([state0[None], rows])

    return trajectory


@pytest.mark.parametrize("scheme", WRONG_SCHEMES)
@pytest.mark.parametrize("problem", PROBLEMS, ids=_problem_id)
def test_n1_wrong_schemes_fail_the_exact_gate_where_theory_says_they_must(
    scheme: str, problem: chk.ExactProblem
) -> None:
    """N1 (class slope band and ceiling): the G1 check fails for wrong schemes, as expected.

    W1 and W3 are typos in a stage and fail every problem. W2 has RK4's stability polynomial
    and passes the two linear problems, which is exactly why G1 has two nonlinear ones: it fails
    them with a slope near 2 (sections 2.4 and 3). The check passes or fails according to the
    theory of the note, not according to anything measured here.

    Catches a G1 that cannot fail, or one that is blind to a wrong stage placement. Blind to
    defects that no problem in the set can see.
    """
    step, expected_to_pass = WRONG_SCHEMES[scheme]
    errors = chk.rk4_exact_errors(problem, trajectory=_trajectory_from_step(step))
    if problem.name in expected_to_pass:
        chk.check_exact_rk4(problem, errors)
    else:
        with pytest.raises(AssertionError):
            chk.check_exact_rk4(problem, errors)


# ---- N2 and N3: wrong parameters (the wiring check and the energy anchor can fail) ----------


def _swap_lengths(p: dp.Params) -> dp.Params:
    return dp.Params(p.m1, p.m2, p.L2, p.L1, p.g)


WRONG_PARAMS = {
    "lengths-swapped": _swap_lengths,
    "g-replaced-by-1": lambda p: dp.Params(p.m1, p.m2, p.L1, p.L2, 1.0),
    "parameters-ignored": lambda p: dp.DEFAULT_PARAMS,
    "masses-swapped": lambda p: dp.Params(p.m2, p.m1, p.L1, p.L2, p.g),
}


def test_n2_swapping_the_lengths_in_rk4_only_fails_the_agreement_check() -> None:
    """N2 (class ceiling): RK4 given ``(L2, L1)`` while the wrapper gets the true parameters.

    Catches a G5 that cannot fail. The measurement is made outside the ``raises`` block, so an
    exception from the implementation is an error and not a pass.
    """
    measured = chk.measure_all_wiring(rk4_params=_swap_lengths)
    worst = max(m.agreement for m in measured)
    with pytest.raises(AssertionError):
        chk.check_at_most(worst, tol.WIRE_AGREE_TOL, what="swapped lengths in RK4 only")


@pytest.mark.parametrize("name", WRONG_PARAMS)
def test_n3_the_same_wrong_parameters_in_both_integrators_pass_agreement_but_fail_the_anchor(
    name: str,
) -> None:
    """N3 (class ceiling): both integrators get the same wrong parameters.

    The two solve the same wrong equations accurately, so they agree: the agreement check of G5
    is blind to this error, which is the reason for the energy anchor. The anchor, evaluated with
    the set's true parameters, sees it in both trajectories. Catches a G6 that cannot fail, and
    shows what G5 alone would have missed.
    """
    wrong = WRONG_PARAMS[name]
    measured = chk.measure_all_wiring(rk4_params=wrong, wrapper_params=wrong)
    chk.check_at_most(
        max(m.agreement for m in measured),
        tol.WIRE_AGREE_TOL,
        what=f"{name} in both: agreement (expected to pass, it is blind to this)",
    )
    for label, drift in (
        ("RK4", max(m.energy_rk4 for m in measured)),
        ("wrapper", max(m.energy_wrapper for m in measured)),
    ):
        with pytest.raises(AssertionError):
            chk.check_at_most(
                drift, tol.WIRE_ENERGY_TOL, what=f"{name} in both: {label} energy drift"
            )
