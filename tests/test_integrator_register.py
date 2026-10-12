"""Tripwire for the milestone 3 register, guards for its seeded cases, and tests of its helpers.

K1 pins the declared values in ``src/horizon/tolerances.py`` (D-044 to D-048). C1 to C3 check
that the seeded cases of ``tests/integrator_checks.py`` follow the recipes of the log, by
recomputing each recipe here from its stream, independently of that module. K2 checks that the
helpers behind every gate fail where they must. The claims and the tests that check them are
tabulated in ``docs/derivations/integrators.md``, section 8. None of these tests needs the
integrators.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from tests import cases
from tests import integrator_checks as chk
from tests.cases import _draw_params


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


# ---- C1 to C3: the seeded cases follow the logged recipes -----------------------------------


def _rest_potential(theta: np.ndarray, params: dp.Params) -> float:
    """``V`` at ``omega = 0``, written out so that it does not depend on ``double_pendulum``."""
    return -(params.m1 + params.m2) * params.g * params.L1 * math.cos(
        theta[0]
    ) - params.m2 * params.g * params.L2 * math.cos(theta[1])


def test_c1_regular_states_follow_the_declared_recipe() -> None:
    """C1 (class exact): 12 states, ``omega = 0``, ``theta`` from stream 2000, reproducible.

    Catches a wrong stream, a wrong draw shape or order, an amplitude other than the declared
    one, and a cache that hides non-reproducibility. Blind to whether the recipe is a good one.
    """
    rng = np.random.default_rng([tol.CASES_SEED, 2000])
    theta = rng.uniform(-0.4, 0.4, size=(12, 2))
    states = chk.regular_states()
    assert states.shape == (12, 4)
    assert np.array_equal(states[:, :2], theta)
    assert np.all(states[:, 2:] == 0.0)
    assert not states.flags.writeable
    assert np.array_equal(states, chk.regular_states.__wrapped__())


def test_c2_chaotic_states_follow_the_declared_recipe() -> None:
    """C2 (class exact): 30 states from stream 2001, per attempt two angles, kept if ``E > 0``.

    With ``omega = 0`` and the default parameters, ``E = V = -2 cos(theta1) - cos(theta2)``,
    written out here. Catches a wrong stream, a wrong acceptance rule, an off-by-one in the
    count, and a state with non-positive energy. Blind to whether ``E > 0`` is a good proxy for
    chaos (it is not a classification; milestone 6).
    """
    rng = np.random.default_rng([tol.CASES_SEED, 2001])
    expected = []
    while len(expected) < 30:
        theta = rng.uniform(-math.pi, math.pi, size=2)
        if _rest_potential(theta, dp.DEFAULT_PARAMS) > 0.0:
            expected.append([theta[0], theta[1], 0.0, 0.0])
    states = chk.chaotic_states()
    assert states.shape == (30, 4)
    assert np.array_equal(states, np.array(expected))
    assert not states.flags.writeable
    assert np.array_equal(states, chk.chaotic_states.__wrapped__())
    energies = [float(dp.energy(jnp.asarray(state), dp.DEFAULT_PARAMS)) for state in states]
    assert min(energies) > 0.0


def test_c3_wiring_cases_follow_the_declared_recipe() -> None:
    """C3 (class exact): 10 sets from stream 2002, 3 states each from stream 2003, set by set.

    The sets come from the D-031 recipe (``tests.cases``), drawn one after the other; the states
    use the acceptance rule of C2 under each set's own parameters. Catches a swapped stream, a
    wrong order of consumption, an energy test with the default parameters instead of the set's,
    and a wrong time unit. Blind to whether the sets are representative.
    """
    param_rng = np.random.default_rng([tol.CASES_SEED, 2002])
    state_rng = np.random.default_rng([tol.CASES_SEED, 2003])
    wiring = chk.wiring_cases()
    assert len(wiring) == 30
    k = 0
    for _ in range(10):
        params = dp.Params(*_draw_params(param_rng))
        assert all(tol.PARAM_LOW <= value <= tol.PARAM_HIGH for value in params)
        assert cases.min_pairwise_separation(params) >= tol.MIN_PARAM_SEPARATION
        kept = 0
        while kept < 3:
            theta = state_rng.uniform(-math.pi, math.pi, size=2)
            if _rest_potential(theta, params) > 0.0:
                case = wiring[k]
                assert case.index == k
                assert case.params == params
                assert np.array_equal(case.state, [theta[0], theta[1], 0.0, 0.0])
                assert case.t_s == math.sqrt(params.L1 / params.g)
                k += 1
                kept += 1
    assert k == 30
    assert np.array_equal(
        [case.state for case in wiring], [case.state for case in chk.wiring_cases.__wrapped__()]
    )


# ---- K2: the check helpers fail where they must --------------------------------------------


def test_k2_check_at_most() -> None:
    """K2 (class exact): ``check_at_most`` passes at the ceiling and fails above it or on NaN.

    Catches a helper that passes a non-finite statistic, and an off-by-one at the boundary.
    Blind to the thresholds themselves (K1).
    """
    assert chk.check_at_most(0.5, 1.0, what="x") == 0.5
    assert chk.check_at_most(1.0, 1.0, what="x") == 1.0
    for bad in (1.0 + 1e-12, math.nan, math.inf, -math.inf):
        with pytest.raises(AssertionError):
            chk.check_at_most(bad, 1.0, what="x")
    for ceiling in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            chk.check_at_most(0.5, ceiling, what="x")


def test_k2_check_at_least() -> None:
    """K2 (class exact): ``check_at_least`` passes at the floor and fails below it or on NaN.

    A floor of zero would pass everything, so it is rejected. Catches the same defects as
    ``check_at_most`` and a vacuous validity rule in G2. Blind to the thresholds (K1).
    """
    assert chk.check_at_least(2.0, 1.0, what="x") == 0.5
    assert chk.check_at_least(1.0, 1.0, what="x") == 1.0
    for bad in (1.0 - 1e-12, math.nan, math.inf, -math.inf):
        with pytest.raises(AssertionError):
            chk.check_at_least(bad, 1.0, what="x")
    for floor in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            chk.check_at_least(2.0, floor, what="x")


def test_k2_check_in_band() -> None:
    """K2 (class exact): ``check_in_band`` passes inside and on the edges, fails outside or on NaN.

    Catches a band applied with the wrong inequality and a non-finite slope that slips through.
    Blind to the bands themselves (K1).
    """
    band = (3.9, 4.1)
    assert chk.check_in_band(4.0, band, what="x") == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < chk.check_in_band(4.05, band, what="x") < 1.0
    chk.check_in_band(3.9, band, what="x")
    chk.check_in_band(4.1, band, what="x")
    for bad in (3.899, 4.101, math.nan, math.inf, -math.inf):
        with pytest.raises(AssertionError):
            chk.check_in_band(bad, band, what="x")
    with pytest.raises(ValueError):
        chk.check_in_band(4.0, (4.1, 3.9), what="x")


def test_k2_log_log_slope_and_grid_spacing() -> None:
    """K2 (class IDENTITY): the slope of an exact power law, and the grid-spacing guard.

    ``error = c dt^p`` has slope ``p`` exactly; a zero or NaN error has no logarithm and gives NaN
    so that the band check rejects it. ``steps_per_unit`` must give integer steps per unit time.
    Catches a slope taken with the wrong base or ordering. Blind to noise in real errors.
    """
    dts = tol.EXACT_DT_GRID
    for order in (1.0, 2.0, 4.0):
        errors = [3.0 * dt**order for dt in dts]
        assert chk.log_log_slope(dts, errors) == pytest.approx(order, abs=1e-12)
    for bad in (0.0, math.nan, math.inf):
        assert math.isnan(chk.log_log_slope(dts, [1e-9, 1e-10, bad, 1e-12]))
    assert [chk.steps_per_unit(dt) for dt in dts] == [25, 50, 100, 200]
    with pytest.raises(ValueError):
        chk.steps_per_unit(0.003)
