"""Smoke tests: the package imports, and importing it alone enables JAX float64.

The float64 checks run in a fresh interpreter: an in-process check could pass only
because ``conftest.py`` already enabled float64 (D-015). They use JAX arrays, never
Python floats, because Python's own ``1.0 + 1e-12 != 1.0`` is always true.
"""

import os
import subprocess
import sys
import textwrap
import warnings

import pytest
from hypothesis import settings

import horizon


def _run_fresh_interpreter(code: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter whose environment cannot already enable float64."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    return subprocess.run(
        [sys.executable, *flags, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_package_imports() -> None:
    assert horizon.__doc__


def test_importing_horizon_alone_enables_float64() -> None:
    result = _run_fresh_interpreter(
        """
        import horizon
        import jax.numpy as jnp

        one = jnp.asarray(1.0)
        assert jnp.zeros(()).dtype == jnp.float64
        assert one.dtype == jnp.float64
        assert jnp.finfo(one.dtype).eps < 1e-15
        assert bool(one + 1e-12 != one)
        """
    )
    assert result.returncode == 0, result.stderr


def test_without_horizon_float64_requests_are_truncated_and_warn() -> None:
    """Negative control: without ``import horizon`` JAX stays in float32 mode.

    It also checks that JAX's real truncation warning still matches the fatal
    ``filterwarnings`` entry in pyproject.toml (D-018). If a JAX upgrade rewords the
    warning, this test fails and the filter must be updated.
    """
    result = _run_fresh_interpreter(
        """
        import jax.numpy as jnp

        jnp.asarray(1.0, dtype=jnp.float64)
        """,
        "-W",
        "error:Explicitly requested dtype:UserWarning",
    )
    assert result.returncode != 0
    assert "Explicitly requested dtype" in result.stderr


def test_dtype_truncation_warning_is_fatal_under_the_pytest_config() -> None:
    with pytest.raises(UserWarning, match="Explicitly requested dtype"):
        warnings.warn(
            "Explicitly requested dtype float64 is not available", UserWarning, stacklevel=2
        )


def test_default_hypothesis_profile_is_deterministic() -> None:
    profile = settings.get_profile("default")
    assert profile.derandomize is True
    assert profile.deadline is None
    assert profile.database is None
