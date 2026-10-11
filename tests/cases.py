"""Frozen test cases: seeded non-default parameter sets and random states (D-031, D-032).

The draw order is part of the frozen design: for parameter set ``k`` the stream is
``np.random.default_rng([CASES_SEED, k])``; it draws the parameters first (log-uniform,
redrawn until all pairs differ by at least ``MIN_PARAM_SEPARATION``), then ``theta`` with
``size=(N_STATES, 2)``, then ``omega`` with ``size=(N_STATES, 2)``. The large-angle states use
their own stream so they do not shift the moderate cases. Reproducibility is tied to the
NumPy version locked in ``uv.lock``.
"""

import functools
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from horizon import tolerances as tol


@dataclass(frozen=True)
class Case:
    """One parameter set with the states drawn for it.

    Attributes
    ----------
    index : int
        Index of the parameter set (0 to ``N_PARAM_SETS - 1``).
    params : tuple of float
        ``(m1, m2, L1, L2, g)``.
    states : numpy.ndarray
        Read-only array of shape ``(n, 4)``, rows ``(theta1, theta2, omega1, omega2)``.
    """

    index: int
    params: tuple[float, float, float, float, float]
    states: np.ndarray


def min_pairwise_separation(params: Sequence[float]) -> float:
    """Return the smallest ``|a - b| / max(a, b)`` over all pairs of parameters."""
    return min(abs(a - b) / max(a, b) for a, b in itertools.combinations(params, 2))


def _draw_params(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    while True:
        values = np.exp(rng.uniform(np.log(tol.PARAM_LOW), np.log(tol.PARAM_HIGH), size=5))
        if min_pairwise_separation(values) >= tol.MIN_PARAM_SEPARATION:
            m1, m2, L1, L2, g = (float(v) for v in values)
            return (m1, m2, L1, L2, g)


def _frozen(states: np.ndarray) -> np.ndarray:
    states.setflags(write=False)
    return states


@functools.cache
def moderate_cases() -> tuple[Case, ...]:
    """Return the ``N_PARAM_SETS`` parameter sets with ``N_STATES`` states each."""
    cases = []
    for k in range(tol.N_PARAM_SETS):
        rng = np.random.default_rng([tol.CASES_SEED, k])
        params = _draw_params(rng)
        theta = rng.uniform(-tol.THETA_MAX, tol.THETA_MAX, size=(tol.N_STATES, 2))
        omega = rng.uniform(-tol.OMEGA_MAX, tol.OMEGA_MAX, size=(tol.N_STATES, 2))
        cases.append(Case(k, params, _frozen(np.hstack([theta, omega]))))
    return tuple(cases)


@functools.cache
def large_angle_cases() -> tuple[Case, ...]:
    """Return the large-angle states (D-032), one per case, on parameter sets 0 to 4."""
    rng = np.random.default_rng([tol.CASES_SEED, tol.LARGE_ANGLE_STREAM])
    n = tol.N_LARGE_ANGLE_STATES
    theta = rng.uniform(-tol.LARGE_THETA_MAX, tol.LARGE_THETA_MAX, size=(n, 2))
    omega = rng.uniform(-tol.OMEGA_MAX, tol.OMEGA_MAX, size=(n, 2))
    sets = moderate_cases()
    return tuple(
        Case(k, sets[k].params, _frozen(np.hstack([theta[k : k + 1], omega[k : k + 1]])))
        for k in range(n)
    )
