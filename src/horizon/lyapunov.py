"""Lyapunov exponents: Benettin's two-trajectory method and the tangent-space method with QR.

The interface is fixed by D-077 and the gates that define "correct" were declared in D-079 to
D-086 before any code existed. The derivations are in ``docs/derivations/lyapunov.md``. The
owner derived the method in the milestone 5 walkthrough (D-039); the function bodies were
implemented by the agent.

Conventions (D-077, after D-028 and D-042)
------------------------------------------
* One state per call, ``float64``, a required ``params``, an autonomous field
  ``f(state, params)``. Batch with ``jax.vmap``.
* Both estimators step with ``integrators.rk4_step`` at a fixed ``dt``. The tangent vectors
  are advanced by ``jax.jvp`` of that step, the exact derivative of the discrete map the state
  follows, so Benettin and QR measure the same map.
* Step counts are static Python ints, checked in Python; ``dt``, the states, the frame and
  ``params`` are traced. Times are integer multiples of ``dt`` and are never accumulated.
* Integrated states are never wrapped (D-010). ``state_difference`` is the only place where an
  angle difference is wrapped.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from horizon.integrators import _require_count, rk4_step

type Field[P] = Callable[[Array, P], Array]
type Step[P] = Callable[[Array, P, Array], Array]
type TangentStep[P] = Callable[[Array, P, Array, Array], Array]


class BenettinResult(NamedTuple):
    """Output of ``benettin``.

    Attributes
    ----------
    log_growth : jax.Array
        ``ln(d / d0)`` of every recorded interval, shape ``(n_renorm,)``.
    final_state : jax.Array
        The reference trajectory's state at the end, shape ``(n,)``.
    max_abs_state : jax.Array
        The largest ``||x(t)||_inf`` of the reference trajectory over every step of the
        recorded window, a scalar (``X`` of ``delta_BQ``, D-082).
    """

    log_growth: Array
    final_state: Array
    max_abs_state: Array


class QRResult(NamedTuple):
    """Output of ``tangent_qr``.

    Attributes
    ----------
    log_r : jax.Array
        ``ln|R_ii|`` of every recorded interval, shape ``(n_qr, k)``.
    states : jax.Array
        The state at each recorded QR time from the end of the transient on, shape
        ``(n_qr + 1, n)``.
    final_state : jax.Array
        The state at the end, shape ``(n,)``.
    max_abs_state : jax.Array
        The largest ``||x(t)||_inf`` over every step of the recorded window, a scalar.
    cond_r : jax.Array
        The 2-norm condition number ``kappa_2(R_j)`` of every recorded interval, shape
        ``(n_qr,)`` (D-079(a)).
    log_det_steps : jax.Array or None
        With ``step_logdet=True``: per recorded interval, the sum over its steps of
        ``ln|det J_step|`` from ``slogdet(jacfwd(rk4_step))``, shape ``(n_qr,)``. Otherwise
        ``None``.
    """

    log_r: Array
    states: Array
    final_state: Array
    max_abs_state: Array
    cond_r: Array
    log_det_steps: Array | None


def wrap(delta: ArrayLike) -> Array:
    """Return ``((delta + pi) mod 2 pi) - pi`` elementwise, in ``[-pi, pi)`` (D-010).

    Parameters
    ----------
    delta : array_like
        Angle differences in radians.

    Returns
    -------
    jax.Array
        The wrapped differences, same shape, ``float64``.
    """
    raise NotImplementedError


def state_difference(a: ArrayLike, b: ArrayLike, angle_mask: tuple[bool, ...]) -> Array:
    """Return ``a - b`` with ``wrap`` applied where the static ``angle_mask`` is true.

    Parameters
    ----------
    a, b : array_like
        States of shape ``(n,)``.
    angle_mask : tuple of bool
        Static, length ``n``: which components are angles.

    Returns
    -------
    jax.Array
        The difference, shape ``(n,)``.
    """
    raise NotImplementedError


def benettin[P](
    f: Field[P],
    state0: ArrayLike,
    params: P,
    direction0: ArrayLike,
    *,
    dt: ArrayLike,
    d0: ArrayLike,
    steps_per_renorm: int,
    n_transient: int,
    n_renorm: int,
    angle_mask: tuple[bool, ...],
) -> BenettinResult:
    """Estimate the largest exponent by Benettin's two-trajectory method (D-077).

    See ``docs/derivations/lyapunov.md`` and D-077 for the method.
    """
    raise NotImplementedError


def tangent_qr[P](
    f: Field[P],
    state0: ArrayLike,
    params: P,
    frame0: ArrayLike,
    *,
    dt: ArrayLike,
    steps_per_qr: int,
    n_transient: int,
    n_qr: int,
    step_logdet: bool = False,
) -> QRResult:
    """Estimate the first ``k`` exponents by the tangent-space method with QR (D-077).

    See ``docs/derivations/lyapunov.md`` and D-077 for the method.
    """
    raise NotImplementedError


def exponents(log_terms: ArrayLike, interval: ArrayLike) -> Array:
    """Return the mean of ``log_terms`` over the recorded intervals, divided by ``interval``."""
    raise NotImplementedError


def running_exponents(log_terms: ArrayLike, interval: ArrayLike) -> Array:
    """Return the cumulative estimate after each recorded interval, shaped like ``log_terms``."""
    raise NotImplementedError


__all__ = [
    "BenettinResult",
    "QRResult",
    "benettin",
    "exponents",
    "running_exponents",
    "state_difference",
    "tangent_qr",
    "wrap",
]

_ = (partial, jax, jnp, _require_count, rk4_step)
