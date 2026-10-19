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

import functools
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
    d = jnp.asarray(delta, dtype=jnp.float64)
    return jnp.mod(d + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def state_difference(a: ArrayLike, b: ArrayLike, angle_mask: tuple[bool, ...]) -> Array:
    """Return ``a - b`` with ``wrap`` applied where the static ``angle_mask`` is true.

    This is the only place where an angle is wrapped (D-077).

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
    diff = jnp.asarray(a, dtype=jnp.float64) - jnp.asarray(b, dtype=jnp.float64)
    return jnp.where(jnp.asarray(angle_mask), wrap(diff), diff)


# ---- the step maps and the seam for the controls (D-077) -----------------------------------
#
# The cores take the step functions as static arguments. They are built once per field and
# cached, so that the same field compiles once.


@functools.cache
def _rk4_step_of[P](f: Field[P]) -> Step[P]:
    """Return the RK4 step map of ``f`` as ``step(state, params, dt)``."""

    def step(state: Array, params: P, dt: Array) -> Array:
        return rk4_step(f, state, params, dt)

    return step


@functools.cache
def _rk4_tangent_of[P](f: Field[P]) -> TangentStep[P]:
    """Return the exact derivative of the RK4 step applied to one vector, by ``jax.jvp``."""

    def tangent(state: Array, params: P, dt: Array, v: Array) -> Array:
        return jax.jvp(lambda y: rk4_step(f, y, params, dt), (state,), (v,))[1]

    return tangent


def _householder(frame: Array) -> tuple[Array, Array]:
    """Re-orthonormalize: the frame becomes ``Q`` of ``jnp.linalg.qr`` and ``R`` is returned."""
    return jnp.linalg.qr(frame)


def _require_transient(value: object) -> None:
    """Raise unless ``value`` is a Python ``int`` of at least 0 (``bool`` is not accepted)."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"n_transient must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"n_transient must be at least 0, got {value}")


def _max_abs(x: Array) -> Array:
    return jnp.max(jnp.abs(x))


# ---- Benettin (D-077) ----------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=("step", "steps_per_renorm", "n_transient", "n_renorm", "angle_mask"),
)
def _benettin_core[P](
    step: Step[P],
    state0: Array,
    params: P,
    direction0: Array,
    dt: Array,
    d0: Array,
    steps_per_renorm: int,
    n_transient: int,
    n_renorm: int,
    angle_mask: tuple[bool, ...],
) -> BenettinResult:
    """Compiled core of ``benettin``; ``step`` advances one state by one step."""

    def one_step(
        carry: tuple[Array, Array, Array], _: None
    ) -> tuple[tuple[Array, Array, Array], None]:
        x, x_tilde, peak = carry
        x = step(x, params, dt)
        x_tilde = step(x_tilde, params, dt)
        return (x, x_tilde, jnp.maximum(peak, _max_abs(x))), None

    def interval(
        carry: tuple[Array, Array, Array], _: None
    ) -> tuple[tuple[Array, Array, Array], Array]:
        (x, x_tilde, peak), _ = jax.lax.scan(one_step, carry, None, length=steps_per_renorm)
        delta = state_difference(x_tilde, x, angle_mask)
        d = jnp.linalg.norm(delta)
        return (x, x + (d0 / d) * delta, peak), jnp.log(d / d0)

    x0 = jnp.asarray(state0, dtype=jnp.float64)
    start = (x0, x0 + d0 * jnp.asarray(direction0, dtype=jnp.float64), jnp.asarray(0.0))
    (x, x_tilde, _), _ = jax.lax.scan(interval, start, None, length=n_transient)
    (x, _, peak), log_growth = jax.lax.scan(
        interval, (x, x_tilde, _max_abs(x)), None, length=n_renorm
    )
    return BenettinResult(log_growth, x, peak)


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

    The reference trajectory starts at ``state0`` and the perturbed one at
    ``state0 + d0 * direction0``. Every ``steps_per_renorm`` steps the method computes
    ``Delta = state_difference(x_tilde, x)`` and ``d = ||Delta||_2``, records ``ln(d / d0)``, and
    sets ``x_tilde = x + (d0 / d) Delta``. The first ``n_transient`` intervals are discarded;
    the next ``n_renorm`` are recorded. The exponent is ``exponents(log_growth, tau_B)`` with
    ``tau_B = steps_per_renorm * dt``.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(state, params)``. Static: pass a module-level function.
    state0 : array_like
        Initial state, shape ``(n,)``.
    params : pytree
        Passed through to ``f``.
    direction0 : array_like
        Unit vector of shape ``(n,)``, the initial direction of the separation.
    dt, d0 : float
        RK4 step and the renormalized separation.
    steps_per_renorm, n_renorm : int
        Python ints of at least 1. Static.
    n_transient : int
        Python int of at least 0. Static.
    angle_mask : tuple of bool
        Static, length ``n``: which components are angles and get wrapped in ``Delta``.

    Returns
    -------
    BenettinResult
        ``log_growth`` of shape ``(n_renorm,)``, ``final_state`` and ``max_abs_state``.

    Raises
    ------
    TypeError, ValueError
        If a count is not an ``int`` or is out of range, or ``angle_mask`` has the wrong length.
    """
    _require_count(steps_per_renorm, "steps_per_renorm")
    _require_count(n_renorm, "n_renorm")
    _require_transient(n_transient)
    x0 = jnp.asarray(state0, dtype=jnp.float64)
    mask = tuple(bool(m) for m in angle_mask)
    if len(mask) != x0.shape[0]:
        raise ValueError(f"angle_mask has {len(mask)} entries for a state of size {x0.shape[0]}")
    return _benettin_core(
        _rk4_step_of(f), x0, params, jnp.asarray(direction0, dtype=jnp.float64), dt, d0,
        steps_per_renorm, n_transient, n_renorm, mask,
    )  # fmt: skip


# ---- QR (D-077) ----------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=(
        "state_step",
        "tangent_step",
        "orthonormalize",
        "steps_per_qr",
        "n_transient",
        "n_qr",
        "step_logdet",
    ),
)
def _tangent_qr_core[P](
    state_step: Step[P],
    tangent_step: TangentStep[P],
    orthonormalize: Callable[[Array], tuple[Array, Array]],
    state0: Array,
    params: P,
    frame0: Array,
    dt: Array,
    steps_per_qr: int,
    n_transient: int,
    n_qr: int,
    step_logdet: bool,
) -> QRResult:
    """Compiled core of ``tangent_qr``, the seam for the controls (D-077).

    ``state_step`` advances the state and ``tangent_step`` one column of the frame;
    ``orthonormalize`` returns the new frame and ``R``. ``log_det_steps`` always follows
    ``state_step``, never ``tangent_step``.
    """
    columns = jax.vmap(tangent_step, in_axes=(None, None, None, 1), out_axes=1)

    def log_det(x: Array) -> Array:
        return jnp.linalg.slogdet(jax.jacfwd(lambda y: state_step(y, params, dt))(x))[1]

    def one_step(
        carry: tuple[Array, Array, Array, Array], _: None
    ) -> tuple[tuple[Array, Array, Array, Array], None]:
        x, frame, peak, acc = carry
        frame = columns(x, params, dt, frame)
        if step_logdet:
            acc = acc + log_det(x)
        x = state_step(x, params, dt)
        return (x, frame, jnp.maximum(peak, _max_abs(x)), acc), None

    def interval(
        carry: tuple[Array, Array, Array], _: None
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array, Array]]:
        x, frame, peak = carry
        (x, frame, peak, acc), _ = jax.lax.scan(
            one_step, (x, frame, peak, jnp.asarray(0.0)), None, length=steps_per_qr
        )
        frame, r = orthonormalize(frame)
        singular = jnp.linalg.svd(r, compute_uv=False)
        return (x, frame, peak), (jnp.log(jnp.abs(jnp.diag(r))), x, singular[0] / singular[-1], acc)

    x0 = jnp.asarray(state0, dtype=jnp.float64)
    frame0 = jnp.asarray(frame0, dtype=jnp.float64)
    (x, frame, _), _ = jax.lax.scan(
        interval, (x0, frame0, jnp.asarray(0.0)), None, length=n_transient
    )
    start = x
    (x, _, peak), (log_r, states, cond_r, acc) = jax.lax.scan(
        interval, (x, frame, _max_abs(x)), None, length=n_qr
    )
    return QRResult(
        log_r,
        jnp.concatenate([start[None], states]),
        x,
        peak,
        cond_r,
        acc if step_logdet else None,
    )


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

    Each step advances the state with ``rk4_step`` and every column of the frame with
    ``jax.jvp`` of ``rk4_step``. Every ``steps_per_qr`` steps the frame is factored as ``QR``;
    ``ln|R_ii|`` is recorded and the frame becomes ``Q``. The first ``n_transient`` intervals
    are discarded; the next ``n_qr`` are recorded. The exponents are
    ``exponents(log_r, tau_Q)`` with ``tau_Q = steps_per_qr * dt``.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(state, params)``. Static: pass a module-level function.
    state0 : array_like
        Initial state, shape ``(n,)``.
    params : pytree
        Passed through to ``f``.
    frame0 : array_like
        ``n x k`` matrix with orthonormal columns.
    dt : float
        RK4 step.
    steps_per_qr, n_qr : int
        Python ints of at least 1. Static.
    n_transient : int
        Python int of at least 0. Static.
    step_logdet : bool, default False
        Static. With ``k = n``, also return ``log_det_steps``: per recorded interval, the sum
        of ``ln|det J_step|`` from ``slogdet(jacfwd(rk4_step))`` along the same trajectory
        (D-079(a)). The frame is never propagated by that matrix.

    Returns
    -------
    QRResult
        See the class.

    Raises
    ------
    TypeError, ValueError
        If a count is not an ``int`` or is out of range, the frame is not ``n x k`` with
        ``k <= n``, or ``step_logdet`` is requested with ``k < n``.
    """
    _require_count(steps_per_qr, "steps_per_qr")
    _require_count(n_qr, "n_qr")
    _require_transient(n_transient)
    x0 = jnp.asarray(state0, dtype=jnp.float64)
    frame = jnp.asarray(frame0, dtype=jnp.float64)
    n = x0.shape[0]
    if frame.ndim != 2 or frame.shape[0] != n or not 1 <= frame.shape[1] <= n:
        raise ValueError(f"frame0 must be n x k with 1 <= k <= n = {n}, got {frame.shape}")
    if step_logdet and frame.shape[1] != n:
        raise ValueError("step_logdet=True needs the full frame, k = n")
    return _tangent_qr_core(
        _rk4_step_of(f), _rk4_tangent_of(f), _householder, x0, params, frame, dt,
        steps_per_qr, n_transient, n_qr, bool(step_logdet),
    )  # fmt: skip


def exponents(log_terms: ArrayLike, interval: ArrayLike) -> Array:
    """Return the mean of ``log_terms`` over the recorded intervals, divided by ``interval``."""
    return jnp.mean(jnp.asarray(log_terms, dtype=jnp.float64), axis=0) / interval


def running_exponents(log_terms: ArrayLike, interval: ArrayLike) -> Array:
    """Return the cumulative estimate after each recorded interval, shaped like ``log_terms``."""
    terms = jnp.asarray(log_terms, dtype=jnp.float64)
    counts = jnp.arange(1, terms.shape[0] + 1, dtype=jnp.float64)
    return jnp.cumsum(terms, axis=0) / (counts.reshape((-1,) + (1,) * (terms.ndim - 1)) * interval)


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
