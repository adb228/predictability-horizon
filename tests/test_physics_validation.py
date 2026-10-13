"""Gates E1, E2, F, V1 to V3 and their negative controls (D-052 to D-060).

The thresholds, states, horizons and seeds were declared in the decision log before any test ran
and live in ``horizon.tolerances`` (section 5 of ``docs/derivations/physics_validation.md``). A
gate that fails is reported, never adjusted (D-025).

The code under test exists, so there is no red phase. The evidence that each gate can fail is the
controls: each runs a deliberately wrong input through the same measurement function and expects
the check to fail because the statistic exceeded its ceiling. The controls match the message
``exceeds the ceiling``, so a statistic that went wrong in another way (a NaN) cannot satisfy
them. Every measurement is made outside the ``raises`` block, so an exception from the code
under test is an error and not a pass.

The measurements are in ``tests/physics_checks.py``, shared with the script
``experiments/stage1/physics_validation.py``, so a gate and the number printed for it are the
same code.
"""

import math

import numpy as np
import pytest

from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from horizon.tolerances import check_close
from tests import integrator_checks as chk
from tests import physics_checks as ph

# The wrong schemes of milestone 3 (W1 to W3, section 2.4 of the integrators note) are reused
# for the controls of V2; they are written there and are not duplicated here.
from tests import test_integrator_gates as m3

P = ph.DEFAULT
EXCEEDS = "exceeds the ceiling"

# ---- E1: DOP853 energy drift (D-053, D-054) -------------------------------------------------

E1_STATES = [pytest.param(s, id=f"regular-{i}") for i, s in enumerate(ph.e1_regular_states())] + [
    pytest.param(s, id=f"chaotic-{i}") for i, s in enumerate(ph.e1_chaotic_states())
]


@pytest.mark.timeout(tol.E_TIMEOUT_S)
@pytest.mark.parametrize("state", E1_STATES)
def test_e1_dop853_energy_drift_stays_under_the_briefs_ceiling(state: np.ndarray) -> None:
    """E1 (class ceiling): ``max_t |E(t) - E(0)| / E_s <= 1e-9`` for DOP853 up to ``t = 1000``.

    DOP853 at its defaults (``rtol = atol = 1e-12``), outputs at every integer time, one test per
    state. The statistic is normalised by the fixed scale ``E_s`` of the parameters (D-053), and
    the ceiling is the brief's number. Catches an energy-dissipating or energy-pumping error in
    the field or the wrapper, and a reference tolerance that is too loose. Blind to an error
    that conserves ``E`` (a consistent time scaling), and to moving states, which the script
    reports and no test gates (D-054).
    """
    energies = ph.dop853_energies(state)
    chk.check_at_most(
        float(ph.drift(energies, P)),
        tol.E_DOP_CEILING,
        what=f"DOP853 energy drift over E_s up to t = {tol.E_HORIZON}",
    )


@pytest.mark.timeout(tol.E_TIMEOUT_S)
@pytest.mark.parametrize(
    "state",
    [
        pytest.param(ph.e1_regular_states()[0], id="regular"),
        pytest.param(ph.e1_chaotic_states()[0], id="chaotic"),
    ],
)
def test_e1_control_a_damped_field_fails_the_ceiling(state: np.ndarray) -> None:
    """E1 control (class ceiling): a term ``-gamma omega`` in the field makes E1 fail.

    ``gamma = 1e-6`` removes energy at a rate far above the ceiling over ``t = 1000``. Catches an
    E1 that cannot fail. Blind to anything that is not dissipative.
    """
    field = ph.damped_field(tol.E_CONTROL_DAMPING)
    value = float(ph.drift(ph.dop853_energies(state, field=field), P))
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(value, tol.E_DOP_CEILING, what="damped field")


@pytest.mark.timeout(tol.E_TIMEOUT_S)
@pytest.mark.parametrize("index", range(tol.E_CONTROL_N_STATES))
def test_e1_control_a_looser_reference_tolerance_fails_the_ceiling(index: int) -> None:
    """E1 control (class ceiling): DOP853 at ``rtol = atol = 1e-10`` fails E1.

    Shows that E1 tells the brief's tolerance from one 100 times looser: the drift scales about
    linearly with the tolerance (D-054). Runs on the first chaotic-ish gate states. Catches an E1
    that cannot see the reference tolerance. Blind to a change smaller than a factor of a few.
    """
    state = ph.e1_chaotic_states()[index]
    value = float(ph.drift(ph.dop853_energies(state, rtol=tol.E_CONTROL_RTOL), P))
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(value, tol.E_DOP_CEILING, what="a looser reference tolerance")


# ---- E2: RK4 energy drift (D-055) ----------------------------------------------------------


def _worst_rk4_drift(states: np.ndarray, dt: float = tol.AGREE_DT) -> float:
    return float(np.max(ph.drift(ph.rk4_energies(states, P, dt), P)))


def test_e2_rk4_energy_drift_stays_under_the_ceiling_on_regular_states() -> None:
    """E2 (class ceiling): RK4 at ``dt = 0.01`` keeps the drift under ``1e-7`` up to ``t = 1000``.

    Twelve regular states. The ceiling comes from the rule of D-055 (ten times the largest of 20
    sample maxima, rounded up). Catches a gross error in a stage or in the step scaling that the
    order check of milestone 3 might miss at a long horizon. Blind to energy-conserving errors.
    """
    chk.check_at_most(
        _worst_rk4_drift(ph.e2_regular_states()),
        tol.E2_CEILING_REGULAR,
        what="RK4 energy drift over E_s, regular states, up to t = 1000",
    )


def test_e2_rk4_energy_drift_stays_under_the_ceiling_on_chaotic_ish_states() -> None:
    """E2 (class ceiling): the same on 30 chaotic-ish states, with its own ceiling, ``5e-4``.

    Catches the same defects in the nonlinear regime. Blind to the same things, and to a method
    of order 3 instead of 4, which G1 and G2 catch.
    """
    chk.check_at_most(
        _worst_rk4_drift(ph.e2_chaotic_states()),
        tol.E2_CEILING_CHAOTIC,
        what="RK4 energy drift over E_s, chaotic-ish states, up to t = 1000",
    )


def test_e2_control_a_coarse_step_fails_the_ceiling() -> None:
    """E2 control (class ceiling): RK4 at ``dt = 0.05`` fails the chaotic-ish ceiling.

    Catches an E2 that cannot fail. Blind to a step only slightly too large.
    """
    value = _worst_rk4_drift(ph.e2_chaotic_states(), tol.E_CONTROL_RK4_DT)
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(value, tol.E2_CEILING_CHAOTIC, what="RK4 at a coarse step")


# ---- F: the small-angle FFT (D-056) --------------------------------------------------------


@pytest.mark.parametrize("integrator", ["dop853", "rk4"])
@pytest.mark.parametrize("mode", list(ph.MODE_DIRECTIONS))
@pytest.mark.parametrize("set_name", list(ph.F_SETS))
def test_f_fft_peaks_match_the_analytic_normal_mode_frequencies(
    set_name: str, mode: str, integrator: str
) -> None:
    """F1, F2 (class relative error): the peaks of ``theta1(t)`` are ``sqrt((g/L)(2 -+ sqrt 2))``.

    ``A = 1e-3``, ``h = 0.05 t_s``, 20000 samples, Hann window, padding, parabolic refinement.
    One mode at a time (the eigenvectors ``(1, +-sqrt 2)``) and the generic state ``(1, 0)``,
    with both integrators and both parameter sets; every peak within ``1e-5`` relative. The
    analytic values are written in ``physics_checks`` from the brief, not from the code under
    test. Catches a wrong frequency, a wrong ``g/L`` or time scaling (F2), and an estimator that
    mislocates the peak. Blind to swapped equal parameters (all of F1 is equal), to errors below
    the tolerance, and to a defect that changes no frequency.
    """
    params = ph.F_SETS[set_name]
    measured = ph.fft_peaks(params, mode, integrator)
    errors = ph.relative_errors(measured, ph.expected_peaks(mode, params))
    chk.check_at_most(max(errors), tol.F_TOL, what=f"{set_name} {mode} {integrator}: peak error")


def test_f_control_a_heavier_second_mass_fails_the_tolerance() -> None:
    """F control (class relative error): ``m2`` scaled by 1.01, against the default frequencies.

    The dynamics are those of a slightly different pendulum while the expected frequencies are
    the default ones. Catches an F that cannot see a one-percent parameter error. Blind to a
    smaller one.
    """
    heavier = dp.Params(P.m1, P.m2 * tol.F_CONTROL_M2_FACTOR, P.L1, P.L2, P.g)
    measured = ph.fft_peaks(heavier, "mode 1", "dop853")
    errors = ph.relative_errors(measured, ph.expected_peaks("mode 1", P))
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(max(errors), tol.F_TOL, what="m2 scaled by 1.01")


def test_f_control_the_wrong_analytic_pair_fails_the_tolerance() -> None:
    """F control (class relative error): comparing with ``2 +- 1`` instead of ``2 +- sqrt 2``.

    Catches an F whose comparison cannot fail. Blind to a pair that is closer to the truth.
    """
    measured = ph.fft_peaks(P, "generic", "dop853")
    wrong = [math.sqrt(2.0 - 1.0), math.sqrt(2.0 + 1.0)]
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(
            max(ph.relative_errors(measured, wrong)), tol.F_TOL, what="the pair 2 +- 1"
        )


# ---- V: reversibility (D-057) --------------------------------------------------------------


def _worst_dop853_round_trip(
    states: np.ndarray, horizon: float, field: ph.Field = dp.vector_field
) -> float:
    return max(ph.dop853_round_trip_error(s, horizon, field=field) for s in states)


def test_v1_dop853_round_trip_returns_to_the_initial_state_on_regular_states() -> None:
    """V1 (class ceiling): forward 100, flip, forward 100, flip back, error at most ``5e-9``.

    Twelve regular states, DOP853 at its defaults. The ceiling comes from the rule of D-055.
    Catches a field that is not time-reversal symmetric (a velocity-odd term), and an integrator
    whose forward and backward legs do not cancel to its tolerance. Blind to a time-reversal
    symmetric error such as a wrong ``g`` (section 4.3 of the note).
    """
    chk.check_at_most(
        _worst_dop853_round_trip(ph.v_regular_states(), tol.V_REGULAR_T),
        tol.V1_CEILING_REGULAR,
        what="DOP853 round trip, regular states",
    )


def test_v1_dop853_round_trip_returns_to_the_initial_state_on_chaotic_ish_states() -> None:
    """V1 (class ceiling): the same on 30 chaotic-ish states over the short horizon 8.

    The error of a round trip grows exponentially with ``T`` in chaotic motion, so the horizon is
    the one of milestone 3's agreement gate (D-057). Ceiling ``5e-8``. Catches the same defects.
    Blind to the same errors, and to anything beyond the horizon.
    """
    chk.check_at_most(
        _worst_dop853_round_trip(ph.v_chaotic_states(), tol.V_CHAOTIC_T),
        tol.V1_CEILING_CHAOTIC,
        what="DOP853 round trip, chaotic-ish states",
    )


def test_v2_rk4_round_trip_returns_to_the_initial_state_on_regular_states() -> None:
    """V2 (class ceiling): the round trip with RK4 at ``dt = 0.01`` on the regular states, ``1e-6``.

    Catches a gross RK4 error at long horizon. Blind to time-reversal symmetric errors.
    """
    chk.check_at_most(
        float(np.max(ph.rk4_round_trip_errors(ph.v_regular_states(), tol.V_REGULAR_T))),
        tol.V2_CEILING_REGULAR,
        what="RK4 round trip, regular states",
    )


def test_v2_rk4_round_trip_returns_to_the_initial_state_on_chaotic_ish_states() -> None:
    """V2 (class ceiling): the round trip with RK4 on the chaotic-ish states at ``T = 8``, ``2e-3``.

    A backstop for gross errors only: this ceiling cannot see a second-order scheme at
    ``dt = 0.01`` (D-058), which G1 and G2 catch.
    """
    chk.check_at_most(
        float(np.max(ph.rk4_round_trip_errors(ph.v_chaotic_states(), tol.V_CHAOTIC_T))),
        tol.V2_CEILING_CHAOTIC,
        what="RK4 round trip, chaotic-ish states",
    )


@pytest.mark.parametrize(
    "states, horizon",
    [
        pytest.param(ph.v_regular_states(), tol.V_CHAOTIC_T, id="regular"),
        pytest.param(ph.v_chaotic_states(), tol.V_CHAOTIC_T, id="chaotic-ish"),
    ],
)
def test_v3_rk4_with_a_negative_step_equals_the_flipped_forward_step(
    states: np.ndarray, horizon: int
) -> None:
    """V3 (class IDENTITY): RK4 with ``-dt`` from ``y`` equals ``S`` RK4 with ``dt`` from ``S y``.

    Exact for a field with ``f(S y) = -S f(y)``, because each stage of one is the sign-flipped
    stage of the other (section 4.2 of the note). It checks D-043's statement that a negative
    ``dt`` is legal and is what a time reversal needs; it is not a second reversibility gate.
    Catches a ``dt`` that is validated, taken through ``abs``, or mishandled in the scan. Blind to
    an error that both sides share.
    """
    negative, flipped = ph.rk4_negative_dt_pair(states, horizon)
    check_close(
        negative,
        flipped,
        rtol=tol.IDENTITY_RTOL,
        atol=tol.IDENTITY_ATOL,
        what="RK4 with a negative step against the flipped forward step",
    )


def test_v1_control_a_damped_field_fails_the_ceiling() -> None:
    """V1 control (class ceiling): the damped field of E1 fails the regular round trip.

    A term ``-gamma omega`` is odd under time reversal, so the round trip cannot return. Runs on
    the first three regular states. Catches a V1 that cannot fail. Blind to a symmetric error.
    """
    field = ph.damped_field(tol.E_CONTROL_DAMPING)
    value = _worst_dop853_round_trip(ph.v_regular_states()[:3], tol.V_REGULAR_T, field=field)
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(value, tol.V1_CEILING_REGULAR, what="damped field")


@pytest.mark.parametrize("scheme", list(m3.WRONG_SCHEMES))
def test_v2_control_wrong_schemes_fail_the_regular_ceiling(scheme: str) -> None:
    """V2 control (class ceiling): W1, W2 and W3 of milestone 3 fail the regular round trip.

    W1 and W3 have a mistyped stage; W2 has RK4's stability polynomial but is second order in
    general (section 2.4 of the integrators note). Catches a V2 that cannot fail on the regular
    states. Blind to a scheme whose error is below the ceiling.
    """
    step, _ = m3.WRONG_SCHEMES[scheme]
    trajectory = m3._trajectory_from_step(step)
    errors = ph.rk4_round_trip_errors(ph.v_regular_states(), tol.V_REGULAR_T, trajectory=trajectory)
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(float(np.max(errors)), tol.V2_CEILING_REGULAR, what=scheme)


def test_v2_control_the_first_order_scheme_fails_the_chaotic_ish_ceiling() -> None:
    """V2 control (class ceiling): W3 (first order) fails the chaotic-ish round trip.

    Only W3: W1 and W2 are second order and stay below this ceiling at ``dt = 0.01`` (D-058), so
    the note records this ensemble as a backstop for gross errors. Catches a V2 that cannot fail
    on the chaotic-ish states. Blind to W1 and W2 by design.
    """
    step, _ = m3.WRONG_SCHEMES["W3-k4-half-step"]
    trajectory = m3._trajectory_from_step(step)
    errors = ph.rk4_round_trip_errors(ph.v_chaotic_states(), tol.V_CHAOTIC_T, trajectory=trajectory)
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(float(np.max(errors)), tol.V2_CEILING_CHAOTIC, what="W3")
