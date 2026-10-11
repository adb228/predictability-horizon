"""Shared pytest configuration.

``horizon`` is imported first so JAX float64 is enabled before any test module creates
an array (docs/DECISIONS.md, D-015). Hypothesis runs are deterministic by default
(D-018); set ``HYPOTHESIS_PROFILE=explore`` for randomized bug-hunting.
"""

import os

from hypothesis import settings

import horizon  # noqa: F401

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
