"""Tripwire for the milestone 3 register and guards for its seeded cases (D-044 to D-048).

The values live in ``src/horizon/tolerances.py``. The claims and the tests that check them are
tabulated in ``docs/derivations/integrators.md``, section 8.
"""

import math

from horizon import tolerances as tol


def test_register_holds_the_declared_integrator_values() -> None:
    """K1 (class exact). Tripwire: these literals are the values logged in D-044 to D-047.

    Catches: an edit to the milestone 3 section of ``src/horizon/tolerances.py`` without a
    matching edit here, so every change shows up in two places in the diff. Changing a value
    needs Opus 5.5's approval first (D-025), and the owner asked that none change after seeing
    results from the implementation.
    Blind to: a declared value that is itself wrong. That is what the review in the log is for.
    """
    # D-044: exact-solution gate.
    assert tol.EXACT_DT_GRID == (0.04, 0.02, 0.01, 0.005)
    assert tol.EXACT_SLOPE_BAND == (3.9, 4.1)
    declared_gates = (
        tol.ExactGate("harmonic", 10, 3e-8),
        tol.ExactGate("linear", 5, 2e-9),
        tol.ExactGate("logistic", 5, 7.5e-12),
        tol.ExactGate("rotation", 5, 1.1e-8),
    )
    assert declared_gates == tol.EXACT_GATES
    assert tol.EXACT_WRAPPER_TOL == 1e-10

    # D-045: pendulum convergence gate.
    assert (tol.CONV_STREAM, tol.N_REGULAR_STATES, tol.REGULAR_THETA_MAX) == (2000, 12, 0.4)
    assert tol.CONV_T == 10
    assert tol.CONV_DT_GRID == (0.04, 0.02, 0.01, 0.005)
    assert (tol.CONV_REFERENCE_TOL, tol.CONV_SELF_CHECK_TOL) == (1e-13, 2.5e-14)
    assert tol.CONV_SLOPE_BAND == (3.8, 4.2)
    assert tol.CONV_FLOOR_RATIO == 30.0

    # D-046: agreement gates.
    assert tol.AGREE_DT == 0.01
    assert (tol.CHAOTIC_STREAM, tol.N_CHAOTIC_STATES) == (2001, 30)
    assert math.pi == tol.CHAOTIC_THETA_MAX
    assert (tol.CHAOTIC_T, tol.CHAOTIC_REFERENCE_TOL, tol.CHAOTIC_TOL) == (8, 1e-12, 1e-4)
    assert (tol.REGULAR_AGREE_T, tol.REGULAR_AGREE_TOL) == (50, 1e-6)

    # D-047: wiring check and energy anchor.
    assert (tol.WIRE_PARAM_STREAM, tol.WIRE_STATE_STREAM) == (2002, 2003)
    assert (tol.N_WIRE_SETS, tol.N_WIRE_STATES) == (10, 3)
    assert (tol.WIRE_DT, tol.WIRE_T, tol.WIRE_REFERENCE_TOL) == (0.005, 2.0, 1e-12)
    assert (tol.WIRE_AGREE_TOL, tol.WIRE_ENERGY_TOL) == (1e-5, 1e-5)
