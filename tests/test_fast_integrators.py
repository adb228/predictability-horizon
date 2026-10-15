"""Reduced versions of the milestone 3 gates G2, G3 and G4, for the fast tier (D-066, D-067).

These are smoke tests, never gates. Each takes the first ``n`` states of its gate's own seeded
ensemble (the same helpers, sliced; no new random stream) and asserts against the same
constants of ``horizon.tolerances`` as the gate: no threshold is new, and none is scaled. The
gate tests in ``tests/test_integrator_gates.py`` are unchanged and run in the full tier; they
are what the claims of milestone 3 rest on. Each test below says what the reduction cannot
catch. Sizes are declared here and pinned by the tripwire at the end of the file.
"""

import numpy as np
import pytest

from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from tests import integrator_checks as chk

G2_N_STATES = 3
"""G2 on the first 3 of the gate's 12 regular states."""
G3_N_STATES = 6
"""G3 on the first 6 of the gate's 30 chaotic-ish states, at the gate's own horizon."""
G4_N_STATES = 3
"""G4 on the first 3 of the gate's 12 regular states."""
G4_HORIZON = 20
"""G4 up to ``t = 20``; the gate runs to ``t = 50`` (``REGULAR_AGREE_T``)."""


def _g2_measurement(states: np.ndarray) -> chk.Convergence:
    """The measurement of ``chk.measure_convergence`` on the given states.

    The helper reads the gate's 12 states internally and has no parameter for them, so its few
    lines are repeated here with the same lower-level helpers and the same constants.
    """
    times = np.arange(tol.CONV_T + 1, dtype=np.float64)
    reference = chk.reference_pendulum(states, dp.DEFAULT_PARAMS, times, tol.CONV_REFERENCE_TOL)
    check_reference = chk.reference_pendulum(
        states, dp.DEFAULT_PARAMS, times, tol.CONV_SELF_CHECK_TOL
    )
    self_errors = np.max(np.abs(reference - check_reference), axis=(1, 2))
    errors = np.empty((len(states), len(tol.CONV_DT_GRID)))
    for j, dt in enumerate(tol.CONV_DT_GRID):
        rows = chk.rk4_pendulum(states, dp.DEFAULT_PARAMS, dt, tol.CONV_T, chk.steps_per_unit(dt))
        errors[:, j] = np.max(np.abs(rows - reference), axis=(1, 2))
    slopes = np.array([chk.log_log_slope(tol.CONV_DT_GRID, row) for row in errors])
    return chk.Convergence(errors, self_errors, slopes)


def test_g2_reduced_rk4_converges_at_order_four_on_the_first_regular_states() -> None:
    """Reduced version of G2 (D-045): 3 of the 12 regular states; same band, same validity rule.

    Per state, the validity rule first (the error at the finest step must be at least
    ``CONV_FLOOR_RATIO`` times the reference's own error), then the slope in ``CONV_SLOPE_BAND``.
    Catches a gross order defect on the pendulum field. Cannot catch a defect that only some of
    the 9 other regular states show, and, with 3 states, gives a slope band that has seen fewer
    draws than the gate's.
    """
    result = _g2_measurement(chk.regular_states()[:G2_N_STATES])
    finest = int(np.argmin(tol.CONV_DT_GRID))
    for i in range(G2_N_STATES):
        chk.check_at_least(
            result.errors[i, finest],
            tol.CONV_FLOOR_RATIO * result.self_errors[i],
            what=f"state {i}: INCONCLUSIVE, error at the finest step against the reference's own",
        )
        chk.check_in_band(
            result.slopes[i], tol.CONV_SLOPE_BAND, what=f"state {i}: convergence slope"
        )


def test_g3_reduced_rk4_and_dop853_agree_on_the_first_chaotic_ish_states() -> None:
    """Reduced version of G3 (D-046): 6 of the 30 chaotic-ish states, the gate's horizon ``T = 8``.

    Same ceiling ``CHAOTIC_TOL``, same reference tolerance. Catches an RK4 or reference defect
    that shows on strongly nonlinear motion. Cannot catch a defect that only the other 24 states
    excite; the worst state of the gate's ensemble lies among these 6 in the calibration, which
    is a fact about this seed and not a guarantee for another change.
    """
    errors = chk.agreement_errors(
        chk.chaotic_states()[:G3_N_STATES], tol.CHAOTIC_T, tol.CHAOTIC_REFERENCE_TOL
    )
    chk.check_at_most(
        float(np.max(errors)),
        tol.CHAOTIC_TOL,
        what=f"chaotic-ish states 0 to {G3_N_STATES - 1}, worst |RK4 - DOP853| up to t = "
        f"{tol.CHAOTIC_T}",
    )


def test_g4_reduced_rk4_and_dop853_agree_on_the_first_regular_states() -> None:
    """Reduced version of G4 (D-046): 3 of the 12 regular states up to ``t = 20``, not ``t = 50``.

    Same ceiling ``REGULAR_AGREE_TOL``, same reference tolerance. Catches a gross phase error of
    RK4 on regular motion. Cannot catch the slow phase drift that needs the gate's long horizon
    (a defect small enough to pass at ``t = 20`` and fail at ``t = 50``), nor one that only the
    other 9 states show.
    """
    errors = chk.agreement_errors(
        chk.regular_states()[:G4_N_STATES], G4_HORIZON, tol.CONV_REFERENCE_TOL
    )
    chk.check_at_most(
        float(np.max(errors)),
        tol.REGULAR_AGREE_TOL,
        what=f"regular states 0 to {G4_N_STATES - 1}, worst |RK4 - DOP853| up to t = {G4_HORIZON}",
    )


# ---- tripwire: the sizes of this file are the ones of D-067 ---------------------------------


def test_the_sizes_of_the_reduced_integrator_gates_are_the_ones_declared_in_d067() -> None:
    assert (G2_N_STATES, G3_N_STATES, G4_N_STATES, G4_HORIZON) == (3, 6, 3, 20)
    # each reduction is smaller than its gate, and takes a prefix of the gate's own ensemble
    assert G2_N_STATES < tol.N_REGULAR_STATES
    assert G3_N_STATES < tol.N_CHAOTIC_STATES
    assert G4_N_STATES < tol.N_REGULAR_STATES
    assert G4_HORIZON < tol.REGULAR_AGREE_T
    assert chk.regular_states()[:G2_N_STATES].shape == (G2_N_STATES, 4)
    assert chk.chaotic_states()[:G3_N_STATES].shape == (G3_N_STATES, 4)


@pytest.mark.parametrize("name", ["G2", "G3", "G4"])
def test_each_reduced_integrator_gate_cites_the_gate_it_reduces(name: str) -> None:
    docstrings = {
        "G2": test_g2_reduced_rk4_converges_at_order_four_on_the_first_regular_states.__doc__,
        "G3": test_g3_reduced_rk4_and_dop853_agree_on_the_first_chaotic_ish_states.__doc__,
        "G4": test_g4_reduced_rk4_and_dop853_agree_on_the_first_regular_states.__doc__,
    }
    doc = docstrings[name] or ""
    assert doc.startswith(f"Reduced version of {name} (D-04")
    assert "Cannot catch" in doc
