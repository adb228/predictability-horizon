"""Reduced versions of the gates E1, V1 and F and their controls, for the fast tier (D-066, D-067).

These are smoke tests, never gates. Each takes the first ``n`` states of its gate's own seeded
ensemble (the same helpers of ``tests/physics_checks.py``, sliced; no new random stream) and
asserts against the same constants of ``horizon.tolerances`` as the gate: no threshold is new,
and none is scaled. The gate tests in ``tests/test_physics_validation.py`` are unchanged and run
in the full tier; the claims of milestone 4 rest on them. Each test says what the reduction
cannot catch, and each reduced control must still fail at the reduced size (D-066 P3).

The measurement helpers of the gates read the gate's size from the register (the E1 horizon, the
F record length) and have no parameter for it, so the few lines that depend on the size are
repeated here with the same lower-level helpers (``solve_reference``, ``energies_along``,
``drift``, ``spectrum_peaks``) and the same constants. ``test_the_reduced_f_measurement_is_the_
gates_measurement`` checks that this copy is the gate's measurement.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from horizon import integrators
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from tests import integrator_checks as chk
from tests import physics_checks as ph

P = ph.DEFAULT
EXCEEDS = "exceeds the ceiling"

E1_HORIZON = 100
"""E1 up to ``t = 100``; the gate runs to ``t = 1000`` (``E_HORIZON``)."""
E1_N_REGULAR = 1
"""The first of the gate's 2 regular states."""
E1_N_CHAOTIC = 1
"""The first of the gate's 5 chaotic-ish states (D-067 knob 2: the second one is dropped)."""
E1_CONTROL_N_STATES = 2
"""The looser-tolerance control runs on the first 2 chaotic-ish states, as in the gate."""
V1_N_REGULAR = 3
"""V1 on the first 3 of the gate's 12 regular states."""
V1_T_REGULAR = 20
"""V1 regular up to ``t = 20``; the gate round-trips to ``t = 100`` (``V_REGULAR_T``)."""
V1_N_CHAOTIC = 6
"""V1 on the first 6 of the gate's 30 chaotic-ish states, at the gate's own horizon ``T = 8``."""
V1_T_CHAOTIC = 8
"""The gate's horizon for the chaotic-ish states (``V_CHAOTIC_T``), not reduced."""
V1_CONTROL_N_REGULAR = 3
"""The damped control of V1 runs on the first 3 regular states, as in the gate."""
F_N_SAMPLES = 4000
"""F on a record of 4000 samples (200 time units); the gate takes 20000 (``F_N_SAMPLES``)."""
F_COPY_CHECK_N_SAMPLES = 1000
"""Record length of the check that the copy of the F measurement equals the gate's (a check on
code, not a measurement, so it is kept short)."""


# ---- E1 (D-053, D-054) ----------------------------------------------------------------------


def _e1_drift(state: np.ndarray, *, field: ph.Field = dp.vector_field, rtol: float | None = None):
    """``max_t |E(t) - E(0)| / E_s`` of one DOP853 run up to ``E1_HORIZON``, at integer times.

    The same steps as ``ph.dop853_energies`` followed by ``ph.drift``; only the horizon differs.
    """
    times = np.arange(E1_HORIZON + 1, dtype=np.float64)
    options = {} if rtol is None else {"rtol": rtol, "atol": rtol}
    rows = integrators.solve_reference(field, state, P, times, **options)
    return float(ph.drift(ph.energies_along(rows, P), P))


E1_STATES = [
    pytest.param(state, id=f"regular-{i}")
    for i, state in enumerate(ph.e1_regular_states()[:E1_N_REGULAR])
] + [
    pytest.param(state, id=f"chaotic-{i}")
    for i, state in enumerate(ph.e1_chaotic_states()[:E1_N_CHAOTIC])
]


@pytest.mark.parametrize("state", E1_STATES)
def test_e1_reduced_dop853_energy_drift_stays_under_the_briefs_ceiling(state: np.ndarray) -> None:
    """Reduced version of E1 (D-054): the same ceiling ``1e-9`` up to ``t = 100``, not ``1000``.

    The first regular state and the first chaotic-ish state of the gate's ensembles, DOP853 at
    its defaults. Catches an energy-dissipating or energy-pumping error in the field or the
    wrapper, and a reference tolerance 100 times looser (its control below). Cannot catch a
    drift that grows slowly and passes at ``t = 100`` but not at ``t = 1000`` (a reference
    tolerance only a few times too loose), nor a defect that only the other states show.
    """
    chk.check_at_most(
        _e1_drift(state),
        tol.E_DOP_CEILING,
        what=f"DOP853 energy drift over E_s up to t = {E1_HORIZON}",
    )


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(ph.e1_regular_states()[0], id="regular"),
        pytest.param(ph.e1_chaotic_states()[0], id="chaotic"),
    ],
)
def test_e1_reduced_control_a_damped_field_fails_the_ceiling(state: np.ndarray) -> None:
    """Reduced control of E1 (D-054): the damped field still fails the same ceiling at ``t = 100``.

    ``gamma = 1e-6`` removes energy far faster than the ceiling allows. Catches a reduced E1 that
    cannot fail. Blind to anything that is not dissipative.
    """
    value = _e1_drift(state, field=ph.damped_field(tol.E_CONTROL_DAMPING))
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(value, tol.E_DOP_CEILING, what="damped field, reduced")


@pytest.mark.parametrize("index", range(E1_CONTROL_N_STATES))
def test_e1_reduced_control_a_looser_reference_tolerance_fails_the_ceiling(index: int) -> None:
    """Reduced control of E1 (D-054): ``rtol = atol = 1e-10`` still fails the ceiling at 100.

    The first chaotic-ish states, as in the gate. ``t = 100`` is the shortest horizon at which
    this control exceeds the ceiling by 3 times (D-067). Catches a reduced E1 that cannot see
    the reference tolerance. Blind to a change smaller than a factor of a few.
    """
    value = _e1_drift(ph.e1_chaotic_states()[index], rtol=tol.E_CONTROL_RTOL)
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(value, tol.E_DOP_CEILING, what="a looser reference tolerance, reduced")


# ---- V1 (D-057) -----------------------------------------------------------------------------


def _worst_round_trip(
    states: np.ndarray, horizon: float, field: ph.Field = dp.vector_field
) -> float:
    return max(ph.dop853_round_trip_error(s, horizon, field=field) for s in states)


def test_v1_reduced_round_trip_returns_to_the_initial_state_on_the_first_regular_states() -> None:
    """Reduced version of V1 (D-057): 3 of the 12 regular states round-tripped to ``t = 20``.

    The same ceiling ``V1_CEILING_REGULAR``. Catches a field that is not time-reversal symmetric
    (a velocity-odd term). Cannot catch an integrator whose forward and backward legs only fail
    to cancel after a longer trip (the gate goes to ``t = 100``), nor a defect that only the
    other 9 states show, and, like the gate, is blind to a time-reversal symmetric error.
    """
    chk.check_at_most(
        _worst_round_trip(ph.v_regular_states()[:V1_N_REGULAR], V1_T_REGULAR),
        tol.V1_CEILING_REGULAR,
        what=f"DOP853 round trip, regular states, reduced to t = {V1_T_REGULAR}",
    )


def test_v1_reduced_round_trip_returns_to_the_initial_state_on_the_first_chaotic_ish_states() -> (
    None
):
    """Reduced version of V1 (D-057): 6 of the 30 chaotic-ish states over the gate's horizon 8.

    The same ceiling ``V1_CEILING_CHAOTIC``. Catches the same defects as the regular test, in the
    nonlinear regime. Cannot catch a defect that only the other 24 states excite, and is blind
    to a time-reversal symmetric error.
    """
    chk.check_at_most(
        _worst_round_trip(ph.v_chaotic_states()[:V1_N_CHAOTIC], V1_T_CHAOTIC),
        tol.V1_CEILING_CHAOTIC,
        what="DOP853 round trip, chaotic-ish states, reduced to 6 states",
    )


def test_v1_reduced_control_a_damped_field_fails_the_ceiling() -> None:
    """Reduced control of V1 (D-057): the damped field still fails the regular ceiling at 20.

    A term ``-gamma omega`` is odd under time reversal, so the round trip cannot return. The first
    3 regular states, as in the gate. Catches a reduced V1 that cannot fail. Blind to a symmetric
    error.
    """
    field = ph.damped_field(tol.E_CONTROL_DAMPING)
    value = _worst_round_trip(ph.v_regular_states()[:V1_CONTROL_N_REGULAR], V1_T_REGULAR, field)
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(value, tol.V1_CEILING_REGULAR, what="damped field, reduced")


# ---- F (D-056) ------------------------------------------------------------------------------


def _fft_peaks(
    params: dp.Params, mode: str, integrator: str, n_samples: int = F_N_SAMPLES
) -> list[float]:
    """``ph.fft_peaks`` for a record of ``n_samples`` samples instead of the gate's 20000.

    The same steps and constants; ``ph.spectrum_peaks`` does the spectrum. The gate's helper
    reads ``tol.F_N_SAMPLES`` inside and has no parameter for it.
    """
    t_s = math.sqrt(params.L1 / params.g)
    h = tol.F_SAMPLE_DT * t_s
    direction = ph.MODE_DIRECTIONS[mode]
    y0 = np.array([tol.F_AMPLITUDE * direction[0], tol.F_AMPLITUDE * direction[1], 0.0, 0.0])
    if integrator == "dop853":
        rows = integrators.solve_reference(dp.vector_field, y0, params, h * np.arange(n_samples))
    elif integrator == "rk4":
        rows = np.asarray(
            integrators.rk4_trajectory(
                dp.vector_field,
                jnp.asarray(y0),
                params,
                tol.AGREE_DT * t_s,
                n_samples - 1,
                tol.F_RK4_STRIDE,
            )
        )
    else:
        raise ValueError(f"unknown integrator {integrator!r}")
    return ph.spectrum_peaks(rows[:, 0], h, ph.PEAK_COUNT[mode])


@pytest.mark.parametrize("integrator", ["dop853", "rk4"])
@pytest.mark.parametrize("mode", list(ph.MODE_DIRECTIONS))
@pytest.mark.parametrize("set_name", list(ph.F_SETS))
def test_f_reduced_fft_peaks_match_the_analytic_normal_mode_frequencies(
    set_name: str, mode: str, integrator: str
) -> None:
    """Reduced version of F (D-056): the same 12 cases and ``F_TOL = 1e-5``, on 4000 samples.

    Both parameter sets, all three initial states, both integrators; every peak within ``1e-5``
    relative of the analytic frequency. The record is 200 time units instead of 1000: a shorter
    record widens the peak, so the error is not smaller at the reduced size (D-066 P3). Catches
    a wrong frequency, a wrong ``g/L`` or time scaling, and an estimator that mislocates the
    peak. Cannot catch a defect that moves a frequency by less than a few times ``1e-6`` (the
    reduced record's own error is about ``2e-6``, an uncommitted calibration) and, like the
    gate, is blind to swapped equal parameters.
    """
    params = ph.F_SETS[set_name]
    errors = ph.relative_errors(
        _fft_peaks(params, mode, integrator), ph.expected_peaks(mode, params)
    )
    chk.check_at_most(
        max(errors), tol.F_TOL, what=f"{set_name} {mode} {integrator}: peak error, reduced"
    )


def test_f_reduced_control_a_heavier_second_mass_fails_the_tolerance() -> None:
    """Reduced control of F (D-056): ``m2`` scaled by 1.01 still fails ``F_TOL`` on 4000 samples.

    The dynamics are those of a slightly different pendulum while the expected frequencies are
    the default ones. Catches a reduced F that cannot see a one-percent parameter error. Blind
    to a smaller one.
    """
    heavier = dp.Params(P.m1, P.m2 * tol.F_CONTROL_M2_FACTOR, P.L1, P.L2, P.g)
    errors = ph.relative_errors(
        _fft_peaks(heavier, "mode 1", "dop853"), ph.expected_peaks("mode 1", P)
    )
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(max(errors), tol.F_TOL, what="m2 scaled by 1.01, reduced")


def test_f_reduced_control_the_wrong_analytic_pair_fails_the_tolerance() -> None:
    """Reduced control of F (D-056): comparing with ``2 +- 1`` still fails on 4000 samples.

    Catches a reduced F whose comparison cannot fail. Blind to a pair that is closer to the truth.
    """
    measured = _fft_peaks(P, "generic", "dop853")
    wrong = [math.sqrt(2.0 - 1.0), math.sqrt(2.0 + 1.0)]
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(
            max(ph.relative_errors(measured, wrong)), tol.F_TOL, what="the pair 2 +- 1, reduced"
        )


@pytest.mark.parametrize("integrator", ["dop853", "rk4"])
def test_the_reduced_f_measurement_is_the_gates_measurement(
    integrator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy ``_fft_peaks`` above returns exactly what ``ph.fft_peaks`` returns.

    A check on the copy, not a measurement: the register's record length is set to a short
    one for this test only (``monkeypatch`` restores it afterwards, and nothing else reads it
    here), and the gate's own helper is run on the same case. The peaks must be identical.
    """
    monkeypatch.setattr(tol, "F_N_SAMPLES", F_COPY_CHECK_N_SAMPLES)
    copy = _fft_peaks(P, "generic", integrator, F_COPY_CHECK_N_SAMPLES)
    assert copy == ph.fft_peaks(P, "generic", integrator)


# ---- tripwire: the sizes of this file are the ones of D-067 ---------------------------------


def test_the_sizes_of_the_reduced_physics_gates_are_the_ones_declared_in_d067() -> None:
    assert (E1_HORIZON, E1_N_REGULAR, E1_N_CHAOTIC, E1_CONTROL_N_STATES) == (100, 1, 1, 2)
    assert (V1_N_REGULAR, V1_T_REGULAR, V1_N_CHAOTIC, V1_T_CHAOTIC) == (3, 20, 6, 8)
    assert V1_CONTROL_N_REGULAR == 3
    assert (F_N_SAMPLES, F_COPY_CHECK_N_SAMPLES) == (4000, 1000)
    # a reduction is not larger than its gate; the chaotic V1 horizon is the gate's own
    assert E1_HORIZON < tol.E_HORIZON
    assert E1_N_REGULAR <= tol.E1_N_REGULAR
    assert E1_N_CHAOTIC <= tol.E1_N_CHAOTIC
    assert E1_CONTROL_N_STATES == tol.E_CONTROL_N_STATES
    assert V1_N_REGULAR <= tol.V_N_REGULAR
    assert V1_T_REGULAR < tol.V_REGULAR_T
    assert V1_N_CHAOTIC <= tol.V_N_CHAOTIC
    assert V1_T_CHAOTIC == tol.V_CHAOTIC_T
    assert V1_CONTROL_N_REGULAR <= V1_N_REGULAR
    assert F_COPY_CHECK_N_SAMPLES < F_N_SAMPLES < tol.F_N_SAMPLES
    # the case lists are as large as declared
    assert len(E1_STATES) == E1_N_REGULAR + E1_N_CHAOTIC


def test_the_reduced_physics_gates_take_the_first_states_of_the_gates_own_ensembles() -> None:
    assert [p.id for p in E1_STATES] == ["regular-0", "chaotic-0"]
    for param, full in zip(E1_STATES[:1], ph.e1_regular_states()[:1], strict=True):
        assert np.array_equal(param.values[0], full)
    for param, full in zip(E1_STATES[1:], ph.e1_chaotic_states()[:E1_N_CHAOTIC], strict=True):
        assert np.array_equal(param.values[0], full)


@pytest.mark.parametrize(
    "function",
    [
        test_e1_reduced_dop853_energy_drift_stays_under_the_briefs_ceiling,
        test_e1_reduced_control_a_damped_field_fails_the_ceiling,
        test_e1_reduced_control_a_looser_reference_tolerance_fails_the_ceiling,
        test_v1_reduced_round_trip_returns_to_the_initial_state_on_the_first_regular_states,
        test_v1_reduced_round_trip_returns_to_the_initial_state_on_the_first_chaotic_ish_states,
        test_v1_reduced_control_a_damped_field_fails_the_ceiling,
        test_f_reduced_fft_peaks_match_the_analytic_normal_mode_frequencies,
        test_f_reduced_control_a_heavier_second_mass_fails_the_tolerance,
        test_f_reduced_control_the_wrong_analytic_pair_fails_the_tolerance,
    ],
    ids=lambda f: f.__name__,
)
def test_each_reduced_physics_test_cites_the_gate_it_reduces_and_says_what_it_cannot_catch(
    function: object,
) -> None:
    doc = " ".join((function.__doc__ or "").split())
    assert doc.startswith(("Reduced version of ", "Reduced control of "))
    assert any(f"{gate} (D-0" in doc[:60] for gate in ("E1", "V1", "F"))
    assert "annot catch" in doc or "Blind to" in doc
