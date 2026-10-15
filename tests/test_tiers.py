"""Guards for the fast and full tiers (docs/DECISIONS.md, D-066 and D-067).

The tiers are assigned by a hook in ``tests/conftest.py`` from the data in ``tests/tiers.py``.
These tests show that the assignment is total, that no rule has gone stale, that ``-m fast`` and
``-m full`` select exactly what ``tier_of`` declares (which also proves that the hook runs before
pytest's own ``-m`` deselection), and that the hook adds nothing but the two markers. The
subprocess runs only collect, about a second each, so no test here needs a timeout override.

What they do not cover: whether a test is in the right tier for its cost (the CI duration guard
looks at that), and anything about how the tiers behave on a runner other than this one.
"""

import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests import conftest, tiers

ROOT = Path(__file__).resolve().parent.parent
TIER_NAMES = ("fast", "full")
PHYSICS = "tests/test_physics_validation.py"
PENDULUM = "tests/test_double_pendulum.py"
ORACLE = "tests/test_oracle_double_pendulum.py"
CASES_PER_SET_FUNCTION = 20
F_FFT = "test_f_fft_peaks_match_the_analytic_normal_mode_frequencies"
NEW_TEST_FILES = {
    "tests/test_tiers.py",
    "tests/test_fast_physics.py",
    "tests/test_fast_integrators.py",
}


def _collect(*extra: str) -> list[str]:
    """The node ids that ``pytest --collect-only -q`` lists for ``tests``, in order."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    done = subprocess.run(
        [*command, *extra, "tests"], cwd=ROOT, capture_output=True, text=True, timeout=45
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return [line for line in done.stdout.splitlines() if line.startswith("tests/") and "::" in line]


@pytest.fixture(scope="module")
def collections() -> dict[str, list[str]]:
    """The three collections, run side by side (each is a separate process of about a second)."""
    selections = {"all": (), "fast": ("-m", "fast"), "full": ("-m", "full")}
    with ThreadPoolExecutor(max_workers=len(selections)) as pool:
        futures = {name: pool.submit(_collect, *extra) for name, extra in selections.items()}
        return {name: future.result() for name, future in futures.items()}


@pytest.fixture
def collected(collections: dict[str, list[str]]) -> list[str]:
    return collections["all"]


@pytest.fixture
def collected_fast(collections: dict[str, list[str]]) -> list[str]:
    return collections["fast"]


@pytest.fixture
def collected_full(collections: dict[str, list[str]]) -> list[str]:
    return collections["full"]


# ---- (a) every item of this session has exactly one tier --------------------------------------


def test_every_collected_item_has_exactly_one_tier_marker_and_it_is_the_declared_one(
    request: pytest.FixtureRequest,
) -> None:
    items = request.session.items
    assert items, "no items in the session"
    wrong = []
    for item in items:
        names = [m.name for m in item.iter_markers() if m.name in TIER_NAMES]
        own = [m.name for m in item.own_markers if m.name in TIER_NAMES]
        expected = [tiers.tier_of(item.nodeid)]
        if names != expected or own != expected:
            wrong.append((item.nodeid, names, own, expected))
    assert not wrong, f"{len(wrong)} items without exactly the declared tier marker: {wrong[:5]}"


# ---- (b) the declared data -------------------------------------------------------------------


def test_the_subset_rule_keeps_the_first_two_of_twenty_sets_fast() -> None:
    assert tiers.MODERATE_FAST_SETS == 2
    declared = tuple(f"set-{i:02d}" for i in range(20))
    assert declared == tiers.SET_IDS
    assert tiers.SET_IDS[0] == "set-00"
    assert tiers.SET_IDS[-1] == "set-19"
    assert len(set(tiers.SET_IDS)) == len(tiers.SET_IDS) == CASES_PER_SET_FUNCTION


def test_the_full_functions_pin_the_counts_of_d067() -> None:
    full = tiers.FULL_FUNCTIONS
    assert all(isinstance(n, int) and n >= 1 for n in full.values())
    by_file = Counter()
    for function, n in full.items():
        by_file[function.split("::")[0]] += n
    assert dict(by_file) == {
        PHYSICS: 28,
        "tests/test_integrator_gates.py": 3,
        ORACLE: 1,
        "tests/test_timeout_config.py": 1,
    }
    assert sum(full.values()) == 33
    assert len(full) == 14
    assert full[f"{PHYSICS}::test_e1_dop853_energy_drift_stays_under_the_briefs_ceiling"] == 7
    assert full[f"{PHYSICS}::{F_FFT}"] == 12


def test_the_subset_functions_are_the_sixteen_per_set_tests_and_the_o2_test() -> None:
    functions = tiers.SUBSET_FUNCTIONS
    assert len(functions) == len(set(functions)) == 17
    assert sum(f.startswith(f"{PENDULUM}::test_t") for f in functions) == 16
    assert [f for f in functions if f.startswith(ORACLE)] == [
        f"{ORACLE}::test_o2_oracle_solution_satisfies_its_euler_lagrange_equations"
    ]
    assert not set(functions) & set(tiers.FULL_FUNCTIONS)


def test_the_number_of_full_tests_is_the_pinned_339() -> None:
    per_family_full = len(tiers.SET_IDS) - tiers.MODERATE_FAST_SETS
    assert sum(tiers.FULL_FUNCTIONS.values()) + len(tiers.SUBSET_FUNCTIONS) * per_family_full == 339


@pytest.mark.parametrize(
    "nodeid, tier",
    [
        (
            f"{PHYSICS}::{F_FFT}[F2-mode 2-dop853]",
            "full",
        ),
        (f"{PHYSICS}::test_f_control_a_heavier_second_mass_fails_the_tolerance", "full"),
        (
            f"{PHYSICS}::test_e1_dop853_energy_drift_stays_under_the_briefs_ceiling[chaotic-4]",
            "full",
        ),
        (f"{PHYSICS}::test_e2_rk4_energy_drift_stays_under_the_ceiling_on_regular_states", "fast"),
        (f"{PENDULUM}::test_t1_mass_matrix_matches_the_oracle[set-00]", "fast"),
        (f"{PENDULUM}::test_t3_acceleration_matches_the_oracle[set-01]", "fast"),
        (f"{PENDULUM}::test_t3_acceleration_matches_the_oracle[set-02]", "full"),
        (f"{PENDULUM}::test_t8_autodiff_jacobian_matches_the_oracle[set-19]", "full"),
        (
            f"{ORACLE}::test_o2_oracle_solution_satisfies_its_euler_lagrange_equations[set-10]",
            "full",
        ),
        (f"{ORACLE}::test_o2_also_holds_at_the_large_angle_states", "fast"),
        (
            f"{PENDULUM}::test_t3_acceleration_matches_the_oracle_at_large_unwrapped_angles[x-5]",
            "fast",
        ),
        (f"{PENDULUM}::test_t3_acceleration_matches_the_oracle[unknown-id]", "fast"),
        (f"{ORACLE}::test_o3_the_derivation_note_equals_the_oracle_symbolically", "full"),
        (
            "tests/test_integrator_gates.py::test_g5_rk4_and_the_wrapper_agree_on_non_default_parameters",
            "fast",
        ),
        (
            "tests/test_integrator_gates.py::test_g2_rk4_converges_at_order_four_on_regular_pendulum_states",
            "full",
        ),
        # a name that only starts like a full function is not that function
        (f"{PHYSICS}::test_f_control_a_heavier_second_mass_fails_the_tolerance_again", "fast"),
        # a parameter id with brackets, '::' and spaces must not confuse the parser
        (f"{PENDULUM}::test_t3_acceleration_matches_the_oracle[a[b]::c d-set-03]", "fast"),
        ("tests/test_fast_physics.py::test_anything[F1-mode 1-dop853]", "fast"),
    ],
)
def test_tier_of_follows_the_rules(nodeid: str, tier: str) -> None:
    assert tiers.tier_of(nodeid) == tier


def test_split_nodeid_handles_brackets_spaces_and_hyphens() -> None:
    nodeid = f"{PHYSICS}::test_f[F2-mode 2-dop853]"
    assert tiers.split_nodeid(nodeid) == (f"{PHYSICS}::test_f", "F2-mode 2-dop853")
    assert tiers.split_nodeid(f"{PHYSICS}::test_f") == (f"{PHYSICS}::test_f", None)


# ---- (c) the rules match the collection exactly ---------------------------------------------


def test_every_full_function_exists_with_its_pinned_number_of_cases(collected: list[str]) -> None:
    counts = Counter(tiers.split_nodeid(nodeid)[0] for nodeid in collected)
    stale = {f: (n, counts[f]) for f, n in tiers.FULL_FUNCTIONS.items() if counts[f] != n}
    assert not stale, f"function: (pinned, collected): {stale}"


def test_every_subset_function_has_the_twenty_set_ids_and_exactly_k_fast_cases(
    collected: list[str],
) -> None:
    for function in tiers.SUBSET_FUNCTIONS:
        cases = [tiers.split_nodeid(n) for n in collected if tiers.split_nodeid(n)[0] == function]
        assert sorted(p for _, p in cases if p is not None) == sorted(tiers.SET_IDS), function
        assert len(cases) == CASES_PER_SET_FUNCTION, function
        tier_counts = Counter(tiers.tier_of(f"{f}[{p}]") for f, p in cases)
        assert tier_counts["fast"] == tiers.MODERATE_FAST_SETS, function
        assert tier_counts["full"] == CASES_PER_SET_FUNCTION - tiers.MODERATE_FAST_SETS, function


def test_the_full_rules_name_no_test_of_the_new_files(collected: list[str]) -> None:
    ours = [n for n in collected if n.split("::")[0] in NEW_TEST_FILES]
    assert ours, "the new test files were not collected"
    assert all(tiers.tier_of(n) == "fast" for n in ours)


# ---- (d) -m fast and -m full select what tier_of declares -------------------------------------


def test_m_fast_and_m_full_select_exactly_the_declared_sets_so_the_hook_runs_before_m(
    collected: list[str], collected_fast: list[str], collected_full: list[str]
) -> None:
    declared_fast = {n for n in collected if tiers.tier_of(n) == "fast"}
    declared_full = {n for n in collected if tiers.tier_of(n) == "full"}
    assert len(collected) == len(set(collected))
    assert set(collected_fast) == declared_fast
    assert set(collected_full) == declared_full
    assert len(collected_fast) == len(declared_fast)
    assert len(collected_full) == len(declared_full)
    # their union is the whole collection, their intersection is empty
    assert set(collected_fast) | set(collected_full) == set(collected)
    assert not set(collected_fast) & set(collected_full)
    # a hook that ran after -m would have selected nothing (no markers yet) or everything
    assert declared_fast and declared_full
    assert len(declared_full) == 339


# ---- (e) the hook adds the two markers and nothing else ---------------------------------------


class FakeItem:
    """An item that records every marker added to it and offers nothing else to the hook."""

    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid
        self.added: list[str] = []

    def add_marker(self, marker: object, append: bool = True) -> None:
        self.added.append(marker.name)  # a MarkDecorator has a name


def test_the_hook_runs_first_and_adds_only_the_tier_marker_without_touching_the_list() -> None:
    hook = conftest.pytest_collection_modifyitems
    assert hook.pytest_impl["tryfirst"] is True
    nodeids = [
        f"{PHYSICS}::{F_FFT}[F2-mode 2-dop853]",
        f"{PENDULUM}::test_t1_mass_matrix_matches_the_oracle[set-00]",
        f"{PENDULUM}::test_t1_mass_matrix_matches_the_oracle[set-07]",
        "tests/test_smoke.py::test_anything",
    ]
    items = [FakeItem(n) for n in nodeids]
    before = list(items)
    hook(config=None, items=items)
    assert len(items) == len(before)
    assert all(a is b for a, b in zip(items, before, strict=True))  # content and order
    assert [i.added for i in items] == [["full"], ["fast"], ["full"], ["fast"]]
    assert [i.nodeid for i in items] == nodeids


# ---- (f) no existing module or helper carries a tier decorator --------------------------------


def test_no_other_test_module_or_helper_mentions_a_tier_marker() -> None:
    allowed = {ROOT / "tests" / "test_tiers.py", ROOT / "tests" / "conftest.py"}
    pattern = re.compile(r"mark\s*\.\s*(fast|full)\b")
    files = sorted((ROOT / "tests").rglob("*.py"))
    assert len(files) >= 15, "the scan found too few files"
    found = [p.name for p in files if p not in allowed and pattern.search(p.read_text())]
    assert not found, f"tier markers written by hand in: {found}"


# ---- (g) the signal-method timeout works inside xdist workers (D-066 Q5) -----------------------

HANG_THEN_CONTROL = """
def test_hangs_forever():
    while True:  # a Python-level hang
        pass


def test_control_runs_after_the_timeout():
    assert True
"""


def test_a_hung_test_fails_with_a_timeout_inside_an_xdist_worker_and_the_session_continues(
    pytester: pytest.Pytester,
) -> None:
    """The CI jobs may run under ``-n auto``: the signal timeout must still fire in a worker.

    The tests run in the worker's main thread, which is why SIGALRM works there. The outer
    ``timeout`` is a hard stop for the subprocess, as in ``tests/test_timeout_config.py``.
    """
    pytester.makepyfile(HANG_THEN_CONTROL)
    result = pytester.runpytest_subprocess(
        "-n", "2", "-v", "--timeout=1", "--timeout-method=signal", "-p", "no:cacheprovider",
        timeout=30,
    )  # fmt: skip
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*Failed: Timeout (>1.0s) from pytest-timeout.*"])
    result.stdout.fnmatch_lines(["[[]gw*FAILED*test_hangs_forever*"])
    result.stdout.fnmatch_lines(["[[]gw*PASSED*test_control_runs_after_the_timeout*"])
