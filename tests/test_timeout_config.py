"""Guard tests for the per-test timeout (docs/DECISIONS.md, D-036).

The timeout is a gate: if it silently stopped working, a bad implementation would hang the
run again. These tests pin its configuration and show, in a subprocess, that it fires and
that the session carries on afterwards. They are the only committed evidence for it.

What they do not cover (D-036): a non-terminating XLA loop, which SIGALRM cannot interrupt,
and a hang at import or collection time. The CI job's own time limit is the backstop.
"""

import pytest

HANG_THEN_CONTROL = """
def test_hangs_forever():
    while True:  # a Python-level hang, as from a `while` loop that never terminates
        pass


def test_control_runs_after_the_timeout():
    assert True
"""


def test_the_timeout_configuration_is_the_one_declared_in_d036(pytestconfig):
    assert float(pytestconfig.getini("timeout")) == 60.0
    assert pytestconfig.getini("timeout_method") == "signal"
    assert pytestconfig.getini("timeout_func_only") is False
    # Diagnostic stack dump only, not a pass/fail threshold; pinned so a silent removal shows.
    assert float(pytestconfig.getini("faulthandler_timeout")) == 120.0


def test_a_hung_test_fails_at_the_limit_and_the_session_continues(pytester: pytest.Pytester):
    pytester.makepyfile(HANG_THEN_CONTROL)
    # The outer `timeout` is a hard stop for the subprocess: if the plugin were broken, this
    # guard would fail at 30 s instead of hanging on the child.
    result = pytester.runpytest_subprocess(
        "--timeout=2", "--timeout-method=signal", "-p", "no:cacheprovider", timeout=30
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*Failed: Timeout (>2.0s) from pytest-timeout.*"])
