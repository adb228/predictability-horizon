"""Tripwire for the milestone 4 register (K1), its seeded-case guards (C) and helper tests (K2).

K1 pins the declared values in ``src/horizon/tolerances.py`` (D-052 to D-060) and the property
that the energy statistic takes its scale from the parameters alone (D-053). C1 to C3 check that
the seeded states of ``tests/physics_checks.py`` follow the recipes of the log, by recomputing
each recipe here from its stream, independently of that module. K2 checks the helpers behind the
gates. The claims and the tests that check them are tabulated in
``docs/derivations/physics_validation.md``, section 6. None of these tests integrates anything.
"""

import math
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from horizon.tolerances import check_close
from tests import integrator_checks as chk
from tests import physics_checks as ph

P = dp.DEFAULT_PARAMS

# ---- K1: the register ---------------------------------------------------------------------


def test_register_holds_the_declared_physics_values() -> None:
    """K1 (class exact). Tripwire: these literals are the values logged in D-052 to D-060.

    Catches: an edit to the milestone 4 section of ``src/horizon/tolerances.py`` without a
    matching edit here, so every change shows up in two places in the diff. Changing a value
    needs Opus 5.5's approval first (D-025). Blind to: a declared value that is itself wrong. That
    is what the review in the log is for.
    """
    # D-054: E1, the DOP853 energy-drift gate, and what is reported beside it.
    assert (tol.E_HORIZON, tol.E_DOP_CEILING) == (1000, 1e-9)
    assert (tol.E1_STREAM_REGULAR, tol.E1_N_REGULAR) == (3000, 2)
    assert (tol.E1_STREAM_CHAOTIC, tol.E1_N_CHAOTIC) == (3001, 5)
    assert tol.E_TIMEOUT_S == 300
    assert tol.E_GROWTH_TIMES == (100, 300, 1000)
    assert tol.E_LITERAL_FLOOR == 0.1
    assert (tol.MOVING_STREAM, tol.N_MOVING, tol.MOVING_OMEGA_MAX) == (3006, 8, 2.0)

    # D-055: E2, the RK4 energy-drift gate.
    assert (tol.E2_STREAM_REGULAR, tol.E2_N_REGULAR, tol.E2_CEILING_REGULAR) == (3002, 12, 1e-7)
    assert (tol.E2_STREAM_CHAOTIC, tol.E2_N_CHAOTIC, tol.E2_CEILING_CHAOTIC) == (3003, 30, 5e-4)

    # D-056: F, the small-angle FFT.
    assert (tol.F_AMPLITUDE, tol.F_SAMPLE_DT, tol.F_N_SAMPLES) == (1e-3, 0.05, 20000)
    assert (tol.F_RK4_STRIDE, tol.F_PAD, tol.F_PEAK_SEPARATION_BINS) == (5, 8, 40)
    assert tol.F_TOL == 1e-5
    assert tol.F2_PARAMS == (1.3, 1.3, 0.7, 0.7, 9.81)

    # D-057: V, time reversibility.
    assert (tol.V_REGULAR_T, tol.V_CHAOTIC_T) == (100, 8)
    assert (tol.V_STREAM_REGULAR, tol.V_N_REGULAR) == (3004, 12)
    assert (tol.V_STREAM_CHAOTIC, tol.V_N_CHAOTIC) == (3005, 30)
    assert (tol.V1_CEILING_REGULAR, tol.V1_CEILING_CHAOTIC) == (5e-9, 5e-8)
    assert (tol.V2_CEILING_REGULAR, tol.V2_CEILING_CHAOTIC) == (1e-6, 2e-3)
    assert tol.V1_REGULAR_TIMEOUT_S == 120

    # D-058: the controls.
    assert (tol.E_CONTROL_RTOL, tol.E_CONTROL_N_STATES) == (1e-10, 2)
    assert (tol.E_CONTROL_DAMPING, tol.E_CONTROL_RK4_DT) == (1e-6, 0.05)
    assert tol.F_CONTROL_M2_FACTOR == 1.01


def test_register_streams_are_distinct_and_disjoint_from_the_earlier_ones() -> None:
    """K1 (class exact): streams 3000 to 3006 are distinct and do not reuse an earlier stream.

    The earlier ones are the parameter-set streams ``k = 0`` to ``19``, 1000 (D-032) and 2000 to
    2003 (D-044 to D-047), as Opus 5.5 checked (D-059). Catches a copy-paste of a stream number,
    which would make two gates draw the same states. Blind to the content of the streams.
    """
    streams = (
        tol.E1_STREAM_REGULAR,
        tol.E1_STREAM_CHAOTIC,
        tol.E2_STREAM_REGULAR,
        tol.E2_STREAM_CHAOTIC,
        tol.V_STREAM_REGULAR,
        tol.V_STREAM_CHAOTIC,
        tol.MOVING_STREAM,
    )
    assert sorted(streams) == list(range(3000, 3007))
    earlier = {*range(20), 1000, 2000, 2001, 2002, 2003}
    assert earlier.isdisjoint(streams)


def test_register_sampling_of_the_fft_record_is_consistent() -> None:
    """K1 (class exact): the stride times the RK4 step is the sampling interval, and the horizon.

    RK4 outputs every ``F_RK4_STRIDE`` steps of ``AGREE_DT``, so the two integrators must sample
    at the same times; the record spans ``T_total = 1000 t_s``. The ``E_GROWTH_TIMES`` are
    increasing and end at the horizon. Catches an edit of one of these numbers that leaves the
    others behind. Blind to whether the numbers are good ones.
    """
    assert abs(tol.F_RK4_STRIDE * tol.AGREE_DT - tol.F_SAMPLE_DT) < 1e-15
    assert abs(tol.F_N_SAMPLES * tol.F_SAMPLE_DT - 1000.0) < 1e-9
    assert list(tol.E_GROWTH_TIMES) == sorted(tol.E_GROWTH_TIMES)
    assert tol.E_GROWTH_TIMES[-1] == tol.E_HORIZON


# A short energy series with a known largest excursion: |E(t) - E(0)| peaks at 2e-3 (at t = 2).
SERIES = np.array([-2.9, -2.9 + 1e-3, -2.9 - 2e-3, -2.9 + 5e-4])
OTHER = dp.Params(2.0, 3.0, 0.5, 0.7, 4.0)


def test_energy_statistic_takes_its_scale_from_the_parameters_alone() -> None:
    """K1 (class exact): ``drift`` divides by ``energy_scale(params)``, whatever the energies are.

    Three consequences of D-053 and its condition (1): the numerator is the largest excursion
    from ``E(0)`` and nothing else; shifting every energy by a constant (moving the zero of the
    potential) changes nothing; and another parameter set changes only the denominator, by
    ``energy_scale``. Catches a normaliser that depends on the state, such as ``|E(0)|`` or
    ``max |E|``, which would break all three. Blind to a wrong ``energy_scale``, which the
    milestone 3 wiring check exercises.
    """
    scale = chk.energy_scale(P)
    assert float(ph.drift(SERIES, P)) == pytest.approx(2e-3 / scale, rel=1e-12)
    assert float(ph.drift(SERIES + 7.5, P)) == pytest.approx(2e-3 / scale, rel=1e-9)
    assert float(ph.drift(SERIES, OTHER)) * chk.energy_scale(OTHER) == pytest.approx(
        2e-3, rel=1e-12
    )
    # The scale is a number of the parameter set: two series with different E(0) share it.
    both = ph.drift(np.stack([SERIES, SERIES * 3.0]), P)
    assert both.shape == (2,)
    assert both[1] == pytest.approx(3.0 * both[0], rel=1e-12)


# ---- C1 to C3: the seeded states follow the logged recipes ----------------------------------


def _rest_potential(theta: np.ndarray) -> float:
    """``V`` of the default parameters at ``omega = 0``, written out, not taken from the code."""
    return -2.0 * math.cos(theta[0]) - math.cos(theta[1])


REGULAR_CASES = [
    pytest.param(tol.E1_STREAM_REGULAR, tol.E1_N_REGULAR, ph.e1_regular_states, id="E1"),
    pytest.param(tol.E2_STREAM_REGULAR, tol.E2_N_REGULAR, ph.e2_regular_states, id="E2"),
    pytest.param(tol.V_STREAM_REGULAR, tol.V_N_REGULAR, ph.v_regular_states, id="V"),
]
CHAOTIC_CASES = [
    pytest.param(tol.E1_STREAM_CHAOTIC, tol.E1_N_CHAOTIC, ph.e1_chaotic_states, id="E1"),
    pytest.param(tol.E2_STREAM_CHAOTIC, tol.E2_N_CHAOTIC, ph.e2_chaotic_states, id="E2"),
    pytest.param(tol.V_STREAM_CHAOTIC, tol.V_N_CHAOTIC, ph.v_chaotic_states, id="V"),
]


@pytest.mark.parametrize("stream, n, function", REGULAR_CASES)
def test_c1_regular_states_follow_the_declared_recipe(
    stream: int, n: int, function: Callable[[], np.ndarray]
) -> None:
    """C1 (class exact): ``n`` states, ``omega = 0``, angles uniform in ``[-0.4, 0.4]``.

    Catches a wrong stream, a wrong draw shape or order, an amplitude other than the declared
    one, and a cache that hides non-reproducibility. Blind to whether the recipe is a good one.
    """
    theta = np.random.default_rng([tol.CASES_SEED, stream]).uniform(-0.4, 0.4, size=(n, 2))
    states = function()
    assert states.shape == (n, 4)
    assert np.array_equal(states[:, :2], theta)
    assert np.all(states[:, 2:] == 0.0)
    assert not states.flags.writeable
    assert np.array_equal(states, function.__wrapped__())  # type: ignore[attr-defined]


@pytest.mark.parametrize("stream, n, function", CHAOTIC_CASES)
def test_c2_chaotic_ish_states_follow_the_declared_recipe(
    stream: int, n: int, function: Callable[[], np.ndarray]
) -> None:
    """C2 (class exact): ``n`` states, per attempt two angles from the stream, kept if ``E > 0``.

    With ``omega = 0`` and the default parameters ``E = V = -2 cos(theta1) - cos(theta2)``,
    written here. Catches a wrong stream, a wrong acceptance rule, an off-by-one in the count,
    and a state with non-positive energy. Blind to whether ``E > 0`` is a good proxy for chaos
    (it is not a classification; milestone 6).
    """
    rng = np.random.default_rng([tol.CASES_SEED, stream])
    expected: list[list[float]] = []
    while len(expected) < n:
        theta = rng.uniform(-math.pi, math.pi, size=2)
        if _rest_potential(theta) > 0.0:
            expected.append([theta[0], theta[1], 0.0, 0.0])
    states = function()
    assert states.shape == (n, 4)
    assert np.array_equal(states, np.array(expected))
    assert not states.flags.writeable
    assert np.array_equal(states, function.__wrapped__())  # type: ignore[attr-defined]
    assert float(np.min(ph.energies_along(states, P))) > 0.0


def test_c3_moving_states_follow_the_declared_recipe() -> None:
    """C3 (class exact): 8 states, one draw of angles in ``[-pi, pi]``, then one of velocities.

    The velocities are uniform in ``[-2, 2]`` (D-054). Catches a swapped order of the two draws,
    a wrong stream, and a wrong velocity bound. Blind to whether the states are representative.
    """
    rng = np.random.default_rng([tol.CASES_SEED, tol.MOVING_STREAM])
    theta = rng.uniform(-math.pi, math.pi, size=(8, 2))
    omega = rng.uniform(-2.0, 2.0, size=(8, 2))
    states = ph.moving_states()
    assert np.array_equal(states, np.hstack([theta, omega]))
    assert not states.flags.writeable
    assert np.array_equal(states, ph.moving_states.__wrapped__())


# ---- K2: the helpers fail where they must ---------------------------------------------------


def test_k2_the_literal_ratio_is_reported_only_above_the_floor() -> None:
    """K2 (class exact): ``literal_ratio`` gives ``|dE| / |E(0)|`` above the floor, else ``None``.

    The floor is ``0.1 E_s`` (0.3 for the defaults, D-053). Catches a floor applied to the wrong
    quantity and a ratio printed where it would only measure ``|E(0)|``. Blind to the choice of
    the floor itself, which D-053 records.
    """
    above = np.array([-0.31, -0.31 + 1e-3, -0.31 - 2e-3])
    below = np.array([0.29, 0.29 + 1e-3, 0.29 - 2e-3])
    assert ph.literal_ratio(above, P) == pytest.approx(2e-3 / 0.31, rel=1e-9)
    assert ph.literal_ratio(below, P) is None


def test_k2_the_peak_finder_recovers_known_frequencies() -> None:
    """K2 (class relative error): two tones are found, in increasing order, without their values.

    A clean two-tone signal on the grid of the FFT gate: both peaks within the gate's own
    tolerance, ``n_peaks = 1`` returns the larger tone only, and the second peak of a single tone
    is more than ``F_PEAK_SEPARATION_BINS`` padded bins from the first. Catches an estimator that
    mislocates a peak, picks one tone twice, or returns a neighbouring maximum of the same peak.
    Blind to the leakage of tones that are closer than a few original bins.
    """
    h = tol.F_SAMPLE_DT
    t = h * np.arange(tol.F_N_SAMPLES)
    w1, w2 = ph.analytic_frequencies(1.0, 1.0)
    two = np.sin(w1 * t + 0.3) + 0.5 * np.sin(w2 * t + 1.1)
    peaks = ph.spectrum_peaks(two, h, 2)
    assert peaks == sorted(peaks)
    assert max(ph.relative_errors(peaks, [w1, w2])) <= tol.F_TOL
    assert max(ph.relative_errors(ph.spectrum_peaks(two, h, 1), [w1])) <= tol.F_TOL

    nfft = 1 << int(np.ceil(np.log2(tol.F_N_SAMPLES * tol.F_PAD)))
    bin_width = 2.0 * np.pi / (nfft * h)
    one = np.sin(w1 * t)
    first, second = ph.spectrum_peaks(one, h, 2)
    assert second - first > tol.F_PEAK_SEPARATION_BINS * bin_width


def test_k2_expected_peaks_refuse_sets_the_formula_does_not_cover() -> None:
    """K2 (class exact): the brief's formula is offered only for equal masses and equal lengths.

    ``expected_peaks`` raises ``ValueError`` when ``m1 != m2`` or ``L1 != L2``, and gives the
    formula for both parameter sets of the F gate. Catches a comparison made against a formula
    that does not apply to the set (D-063). Blind to a set with equal masses and lengths for
    which the caller uses the wrong ``g``.
    """
    with pytest.raises(ValueError):
        ph.expected_peaks("mode 1", dp.Params(1.0, 1.01, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        ph.expected_peaks("mode 1", dp.Params(1.0, 1.0, 1.0, 1.2, 1.0))
    for params in ph.F_SETS.values():
        low, high = ph.analytic_frequencies(params.g, params.L1)
        assert ph.expected_peaks("generic", params) == [low, high]
        assert ph.expected_peaks("mode 1", params) == [low]
        assert ph.expected_peaks("mode 2", params) == [high]


def test_k2_the_analytic_frequencies_are_the_briefs_formula() -> None:
    """K2 (class IDENTITY): ``omega^2 = (g/L)(2 -+ sqrt 2)``, and it scales with ``sqrt(g/L)``.

    Catches a typo in the helper that every F test relies on. Blind to the brief's formula
    being wrong, which milestone 2's T9 and O4 check from the physics.
    """
    low, high = ph.analytic_frequencies(1.0, 1.0)
    check_close(
        np.array([low**2, high**2]),
        np.array([2.0 - math.sqrt(2.0), 2.0 + math.sqrt(2.0)]),
        rtol=tol.IDENTITY_RTOL,
        atol=tol.IDENTITY_ATOL,
        what="the default normal-mode frequencies squared",
    )
    scaled = ph.analytic_frequencies(9.81, 0.7)
    check_close(
        np.array(scaled),
        math.sqrt(9.81 / 0.7) * np.array([low, high]),
        rtol=tol.IDENTITY_RTOL,
        atol=tol.IDENTITY_ATOL,
        what="the scaling with sqrt(g/L)",
    )


def test_k2_the_damped_field_adds_only_the_damping_term() -> None:
    """K2 (class IDENTITY): the damped field minus the field is ``(0, 0, -g omega1, -g omega2)``.

    The control of E1 and V1 must differ from the real field by exactly that. Catches a control
    that also changes the conservative part. Blind to the size of ``gamma``, which D-058 declares.
    """
    state = np.array([0.7, -1.1, 0.4, -0.9])
    gamma = tol.E_CONTROL_DAMPING
    difference = ph.damped_field(gamma)(jnp.asarray(state), P) - dp.vector_field(
        jnp.asarray(state), P
    )
    check_close(
        np.asarray(difference),
        np.array([0.0, 0.0, -gamma * state[2], -gamma * state[3]]),
        rtol=tol.IDENTITY_RTOL,
        atol=tol.IDENTITY_ATOL,
        what="damped field minus field",
    )


def test_k2_the_flip_and_the_summary() -> None:
    """K2 (class exact): ``S`` flips the velocities only and is an involution; ``summary`` orders.

    Catches a flip that touches an angle, and a summary that returns median and minimum swapped.
    Blind to nothing else in the module.
    """
    state = np.array([0.7, -1.1, 0.4, -0.9])
    assert np.array_equal(state * ph.FLIP, np.array([0.7, -1.1, -0.4, 0.9]))
    assert np.array_equal(state * ph.FLIP * ph.FLIP, state)
    assert ph.summary(np.array([3.0, 1.0, 2.0, 5.0])) == (5.0, 2.5, 1.0)
