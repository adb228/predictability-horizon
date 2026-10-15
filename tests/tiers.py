"""The test tiers as data (docs/DECISIONS.md, D-066 and D-067).

Pure Python and no ``pytest`` import, so the guards in ``tests/test_tiers.py`` and the hook in
``tests/conftest.py`` can both read it. Every test belongs to exactly one tier. ``fast`` is the
default, so a forgotten test costs time and not coverage; the heavy existing tests are named here
as ``full`` and the gate test files are not edited to say so. The fast tier is a smoke tier and
is never a gate: the full tier runs everything.

A rule names a test function by its exact node id, ``file::function``, and pins the number of
collected cases, so a renamed, added or removed case fails the guard instead of leaving a stale
rule. Rules never use globs.
"""

from typing import Literal

type Tier = Literal["fast", "full"]

_PHYSICS = "tests/test_physics_validation.py"
_GATES = "tests/test_integrator_gates.py"
_ORACLE = "tests/test_oracle_double_pendulum.py"
_PENDULUM = "tests/test_double_pendulum.py"
_TIMEOUT = "tests/test_timeout_config.py"

FULL_FUNCTIONS: dict[str, int] = {
    # E1, V1 and F (and their controls) are full in their entirety, because each has a reduced
    # fast version in tests/test_fast_physics.py (D-067): 28 tests.
    f"{_PHYSICS}::test_e1_dop853_energy_drift_stays_under_the_briefs_ceiling": 7,
    f"{_PHYSICS}::test_e1_control_a_damped_field_fails_the_ceiling": 2,
    f"{_PHYSICS}::test_e1_control_a_looser_reference_tolerance_fails_the_ceiling": 2,
    f"{_PHYSICS}::test_v1_dop853_round_trip_returns_to_the_initial_state_on_regular_states": 1,
    f"{_PHYSICS}::test_v1_dop853_round_trip_returns_to_the_initial_state_on_chaotic_ish_states": 1,
    f"{_PHYSICS}::test_v1_control_a_damped_field_fails_the_ceiling": 1,
    f"{_PHYSICS}::test_f_fft_peaks_match_the_analytic_normal_mode_frequencies": 12,
    f"{_PHYSICS}::test_f_control_the_wrong_analytic_pair_fails_the_tolerance": 1,
    f"{_PHYSICS}::test_f_control_a_heavier_second_mass_fails_the_tolerance": 1,
    # G2 to G4, reduced in tests/test_fast_integrators.py: 3 tests.
    f"{_GATES}::test_g2_rk4_converges_at_order_four_on_regular_pendulum_states": 1,
    f"{_GATES}::test_g3_rk4_and_dop853_agree_on_chaotic_ish_states_over_the_declared_horizon": 1,
    f"{_GATES}::test_g4_rk4_and_dop853_agree_on_regular_states_over_the_declared_horizon": 1,
    # Two tests of 2 s or more that are not reduced (D-067).
    f"{_ORACLE}::test_o3_the_derivation_note_equals_the_oracle_symbolically": 1,
    f"{_TIMEOUT}::test_a_hung_test_fails_at_the_limit_and_the_session_continues": 1,
}
"""Node id of a full function -> the number of cases it must have when collected."""

MODERATE_FAST_SETS = 3
"""``K``: of the 20 moderate parameter sets, the first ``K`` stay fast and the rest are full."""

SET_IDS: tuple[str, ...] = tuple(f"set-{i:02d}" for i in range(20))
"""The ids of the 20 moderate parameter sets, in the order of their declaration."""

SUBSET_FUNCTIONS: tuple[str, ...] = (
    f"{_PENDULUM}::test_t1_mass_matrix_matches_the_oracle",
    f"{_PENDULUM}::test_t2_energy_matches_the_oracle",
    f"{_PENDULUM}::test_t2_kinetic_energy_matches_the_oracle",
    f"{_PENDULUM}::test_t2_potential_energy_matches_the_oracle",
    f"{_PENDULUM}::test_t3_acceleration_matches_the_oracle",
    f"{_PENDULUM}::test_t4_euler_lagrange_residual_of_the_acceleration_vanishes",
    f"{_PENDULUM}::test_t5_energy_is_stationary_along_the_vector_field",
    f"{_PENDULUM}::test_t6_acceleration_is_even_in_omega_and_odd_in_theta",
    f"{_PENDULUM}::test_t6_vector_field_is_2pi_periodic_in_each_angle",
    f"{_PENDULUM}::test_t7_energy_is_bounded_below_by_the_hanging_energy",
    f"{_PENDULUM}::test_t7_hanging_energy_is_minus_the_gravity_scale",
    f"{_PENDULUM}::test_t7_the_four_equilibria_have_zero_vector_field",
    f"{_PENDULUM}::test_t8_autodiff_jacobian_matches_the_oracle",
    f"{_PENDULUM}::test_t10_gravity_time_scaling_multiplies_acceleration_by_k",
    f"{_PENDULUM}::test_t10_scaling_both_masses_keeps_acceleration_and_scales_energy_linearly",
    f"{_PENDULUM}::test_t10_scaling_lengths_and_gravity_keeps_acceleration_and_scales_energy_by_c_squared",
    f"{_ORACLE}::test_o2_oracle_solution_satisfies_its_euler_lagrange_equations",
)
"""The 16 per-set functions of ``test_double_pendulum.py`` and ``test_o2_*`` of the oracle file.

Each has exactly one case per id of ``SET_IDS``. Only the ids from ``SET_IDS[K:]`` are full; the
large-angle cases and the singletons of the same files are not listed and stay fast."""


def split_nodeid(nodeid: str) -> tuple[str, str | None]:
    """Split ``file::function[params]`` into ``(file::function, params)``.

    ``params`` is ``None`` for an unparametrized test. The parameter ids may hold brackets,
    spaces, hyphens and ``::``, so the split is at the first ``[`` and the last ``]``.
    """
    start = nodeid.find("[")
    if start < 0:
        return nodeid, None
    end = nodeid.rfind("]")
    return nodeid[:start], nodeid[start + 1 : end if end > start else len(nodeid)]


def tier_of(nodeid: str) -> Tier:
    """The tier of one collected test: ``"full"`` if a rule above names it, else ``"fast"``."""
    function, params = split_nodeid(nodeid)
    if function in FULL_FUNCTIONS:
        return "full"
    if function in SUBSET_FUNCTIONS and params in SET_IDS[MODERATE_FAST_SETS:]:
        return "full"
    return "fast"
