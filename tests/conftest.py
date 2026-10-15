"""Shared pytest configuration.

``horizon`` is imported first so JAX float64 is enabled before any test module creates
an array (docs/DECISIONS.md, D-015). Hypothesis runs are deterministic by default
(D-018); set ``HYPOTHESIS_PROFILE=explore`` for randomized bug-hunting.

The test tiers (D-066, D-067) are assigned here at collection from ``tests/tiers.py``: the
hook adds one marker, ``fast`` or ``full``, to every test and does nothing else.
"""

import os

import pytest
from hypothesis import settings

import horizon  # noqa: F401
from tests.tiers import tier_of

# ``pytester`` runs pytest in a subprocess for the timeout guard test (D-036). It must be
# declared here, in the top-level conftest.
pytest_plugins = ["pytester"]

settings.register_profile(
    "default",
    derandomize=True,
    deadline=None,  # the first call of a JIT-compiled function is slow
    database=None,
    print_blob=True,
)
settings.register_profile(
    "explore",
    derandomize=False,
    deadline=None,
    max_examples=1000,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test ``fast`` or ``full`` by ``tier_of(item.nodeid)`` and change nothing else.

    ``tryfirst`` makes this run before pytest's own ``-m`` deselection, so that ``-m fast`` and
    ``-m full`` see the markers. No skip, xfail, timeout, deselection or reordering is added
    here (D-066 P1); ``tests/test_tiers.py`` checks all of that.
    """
    for item in items:
        item.add_marker(getattr(pytest.mark, tier_of(item.nodeid)))
