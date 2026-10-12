"""Fixed-step RK4 in JAX and the SciPy DOP853 reference solver.

The interface is fixed by D-042 and D-043, and the gates that define "correct" were declared
in D-044 to D-047 before any code existed. The scheme, the solver and the limits on how far two
integrators can agree are written out in ``docs/derivations/integrators.md``. The function
bodies were implemented by the agent after the walkthrough (D-039, D-041).

Conventions (D-042)
-------------------
* The system is autonomous: ``dy/dt = f(y, params)``. ``f`` takes the state, an array of shape
  ``(n,)``, and ``params`` (any JAX pytree, ``None`` if there are none) and returns an array of
  shape ``(n,)``. It must be a pure JAX function with no Python branching on values.
* Time has the units that ``f`` uses. For the double pendulum that is ``sqrt(L1 / g)``
  (section 8 of ``docs/derivations/double_pendulum.md``), which is 1 for the default
  parameters. No function here takes or returns a times array: output times follow from
  ``dt``, ``stride`` and the row index.
* One state is integrated per call (D-028). Batch with ``jax.vmap``, over states or over a
  ``Params`` whose fields are arrays.
* Integrated angles are never wrapped (D-010); nothing here knows what the state means.
"""

import warnings
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


def rk4_step[P](
    f: Callable[[Array, P], Array], state: ArrayLike, params: P, dt: ArrayLike
) -> Array:
    """Advance one state by one step of the classical fourth-order Runge-Kutta method.

    With ``h = dt``: ``k1 = f(y)``, ``k2 = f(y + h/2 k1)``, ``k3 = f(y + h/2 k2)``,
    ``k4 = f(y + h k3)`` and ``y + h/6 (k1 + 2 k2 + 2 k3 + k4)`` (section 2.1 of
    ``docs/derivations/integrators.md``). There is no check on ``dt``: a negative step is legal
    and integrates backward in time, which the reversibility tests of milestone 4 need.

    The function is not compiled by itself. It works eagerly, under ``jax.jit`` (with ``f``
    static) and under ``jax.vmap``; to call it many times from Python, wrap it in ``jax.jit``
    or use ``rk4_trajectory``.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(state, params) -> array`` of shape ``(n,)``.
    state : array_like
        The state ``y``, shape ``(n,)``.
    params : pytree
        Passed through to ``f``. May be batched with ``jax.vmap``.
    dt : float
        Step ``h`` in the time unit of ``f``. May be batched too.

    Returns
    -------
    jax.Array
        The state after one step, shape ``(n,)``, ``float64``.
    """
    y = jnp.asarray(state, dtype=jnp.float64)
    k1 = f(y, params)
    k2 = f(y + 0.5 * dt * k1, params)
    k3 = f(y + 0.5 * dt * k2, params)
    k4 = f(y + dt * k3, params)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _require_count(value: object, name: str) -> None:
    """Raise unless ``value`` is a Python ``int`` of at least 1 (``bool`` is not accepted)."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


@partial(jax.jit, static_argnames=("f", "n_out", "stride"))
def _rk4_trajectory[P](
    f: Callable[[Array, P], Array],
    state0: ArrayLike,
    params: P,
    dt: ArrayLike,
    n_out: int,
    stride: int,
) -> Array:
    """Compiled core of ``rk4_trajectory``: two nested scans, only the output rows are stored."""
    y0 = jnp.asarray(state0, dtype=jnp.float64)

    def inner(y: Array, _: None) -> tuple[Array, None]:
        return rk4_step(f, y, params, dt), None

    def outer(y: Array, _: None) -> tuple[Array, Array]:
        y, _ = jax.lax.scan(inner, y, None, length=stride)
        return y, y

    _, rows = jax.lax.scan(outer, y0, None, length=n_out)
    return jnp.concatenate([y0[None], rows])


def rk4_trajectory[P](
    f: Callable[[Array, P], Array],
    state0: ArrayLike,
    params: P,
    dt: ArrayLike,
    n_out: int,
    stride: int,
) -> Array:
    """Integrate with RK4 and keep the state every ``stride`` steps.

    Row ``k`` of the result is the state at time ``k * stride * dt``, counted from the initial
    state, so row 0 is ``state0`` and the last row is the state after ``n_out * stride`` steps.
    Times are integer multiples of ``stride * dt`` and are never accumulated. The loop is two
    nested ``jax.lax.scan``: the outer one over the ``n_out`` outputs and the inner one over the
    ``stride`` steps between them. A negative ``dt`` is legal (see ``rk4_step``).

    ``f``, ``n_out`` and ``stride`` are static: a new function object (for example a new
    ``lambda`` or ``functools.partial`` on every call) or a new count compiles again, so pass a
    module-level function. ``state0``, ``params`` and ``dt`` are traced. The function is
    compiled with ``jax.jit`` and works under ``jax.vmap`` over those three.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(state, params) -> array`` of shape ``(n,)``. Static.
    state0 : array_like
        Initial state, shape ``(n,)``.
    params : pytree
        Passed through to ``f``.
    dt : float
        Step in the time unit of ``f``.
    n_out : int
        Number of output rows after the initial one. A Python ``int`` of at least 1. Static.
    stride : int
        Number of RK4 steps between two output rows. A Python ``int`` of at least 1. Static.

    Returns
    -------
    jax.Array
        Shape ``(n_out + 1, n)``, ``float64``. Row ``k`` is the state at ``t = k * stride * dt``.

    Raises
    ------
    TypeError
        If ``n_out`` or ``stride`` is not an ``int``.
    ValueError
        If ``n_out`` or ``stride`` is less than 1. Zero steps would otherwise return the
        initial state in every row without any warning.
    """
    _require_count(n_out, "n_out")
    _require_count(stride, "stride")
    return _rk4_trajectory(f, state0, params, dt, n_out, stride)


def solve_reference[P](
    f: Callable[[Array, P], Array],
    state0: ArrayLike,
    params: P,
    t_eval: ArrayLike,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Integrate with SciPy's DOP853 and return the solution at the times ``t_eval``.

    This is the reference the fast integrator is checked against (section 4 of
    ``docs/derivations/integrators.md``). ``rtol`` and ``atol`` bound the local error per step,
    not the global error, so in chaotic motion the result is accurate pointwise over a limited
    horizon only (section 5 of the note).

    ``f`` is compiled once with ``jax.jit`` and adapted to SciPy's ``f(t, y)`` signature. The
    integration runs over ``(0, t_eval[-1])``. Any warning raised during the solve becomes an
    exception, so that SciPy cannot silently change a tolerance: SciPy raises an ``rtol`` below
    ``100 * eps`` (about ``2.2e-14``) to that value and only warns.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(state, params) -> array`` of shape ``(n,)``.
    state0 : array_like
        Initial state at ``t = 0``, shape ``(n,)``.
    params : pytree
        Passed through to ``f``.
    t_eval : array_like
        Output times in the time unit of ``f``: one-dimensional, non-empty, with
        ``t_eval[0] >= 0`` and strictly increasing.
    rtol : float, keyword-only, default 1e-12
        Relative tolerance of the step-size control.
    atol : float, keyword-only, default 1e-12
        Absolute tolerance of the step-size control.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(t_eval), n)``, ``float64``. Row ``i`` is the state at ``t_eval[i]``.

    Raises
    ------
    ValueError
        If ``t_eval`` is empty, starts below 0, or is not strictly increasing.
    RuntimeError
        If SciPy reports a failed integration (a non-zero status), for example a step size
        that underflows near a singularity.
    Warning
        Any warning SciPy emits, raised as an exception (for example a ``UserWarning`` for an
        ``rtol`` below ``100 * eps``).
    """
    times = np.asarray(t_eval, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"t_eval must be a non-empty one-dimensional array, got {times.shape}")
    if not np.all(np.isfinite(times)):
        raise ValueError("t_eval must contain only finite times")
    if times[0] < 0.0:
        raise ValueError(f"t_eval[0] must be at least 0, got {times[0]}")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("t_eval must be strictly increasing")

    compiled = jax.jit(f)

    def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(compiled(jnp.asarray(y), params), dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # SciPy clamps a too-small rtol and only warns
        solution = solve_ivp(
            rhs,
            (0.0, float(times[-1])),
            np.asarray(state0, dtype=np.float64),
            method="DOP853",
            t_eval=times,
            rtol=rtol,
            atol=atol,
        )
    if solution.status != 0:
        raise RuntimeError(f"DOP853 stopped with status {solution.status}: {solution.message}")
    return np.asarray(solution.y.T, dtype=np.float64)
