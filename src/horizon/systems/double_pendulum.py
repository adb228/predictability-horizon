"""Planar double pendulum: mass matrix, energies and equations of motion.

The interface is fixed by D-028 and the function bodies are written by the project owner
(D-003). The equations are derived in ``docs/derivations/double_pendulum.md``, and the frozen
tests in ``tests/test_double_pendulum.py`` define "correct" (D-029 to D-034).

Conventions (D-028)
-------------------
* State: array of shape ``(4,)``, ordered ``(theta1, theta2, omega1, omega2)``.
* Both angles are in radians, measured **from the downward vertical** (``theta = 0`` hangs
  straight down, ``theta2`` is *not* relative to rod 1). Angular velocities are in rad per
  unit time. Integrated angles are never wrapped (D-010), so inputs may be far outside
  ``[-pi, pi]``.
* Every function acts on **one** state. Batching is done by the caller with ``jax.vmap``.
* ``params`` is a **required** argument of every function, with no default, so that a test
  or an experiment cannot silently run on the all-ones parameters, which hide swapped-parameter
  bugs.
* Functions are pure, ``float64``, and work under ``jit``, ``vmap`` and ``jacfwd`` with
  ``params`` passed as a traced argument. Do not branch in Python on a value of ``state`` or
  ``params``, and do not close over them.
* Default units are nondimensional: ``m1 = m2 = L1 = L2 = g = 1`` with time measured in units
  of ``sqrt(L1 / g)`` (section 8 of the derivation).
"""

from typing import NamedTuple

from jax import Array


class Params(NamedTuple):
    """Physical parameters of the double pendulum.

    A ``NamedTuple`` of numbers is already a JAX pytree, so a ``Params`` whose fields are
    arrays can be passed to ``jax.vmap`` to batch over parameter sets.

    Attributes
    ----------
    m1 : float
        Mass of the first bob, in mass units.
    m2 : float
        Mass of the second bob.
    L1 : float
        Length of the first rod, in length units.
    L2 : float
        Length of the second rod.
    g : float
        Gravitational acceleration, in length per time squared.
    """

    m1: float = 1.0
    m2: float = 1.0
    L1: float = 1.0
    L2: float = 1.0
    g: float = 1.0


DEFAULT_PARAMS = Params()
"""The all-ones parameters: the nondimensional default (``mu = ell = 1`` in section 8)."""


def mass_matrix(theta: Array, params: Params) -> Array:
    """Return the mass matrix ``M(theta)`` of the equations ``M alpha = b``.

    See section 3 of ``docs/derivations/double_pendulum.md``.

    Parameters
    ----------
    theta : jax.Array
        The two angles ``(theta1, theta2)`` in radians, shape ``(2,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        Symmetric positive-definite matrix, shape ``(2, 2)``, in units of mass times length
        squared. It depends on the angles only through ``theta1 - theta2``.
    """
    raise NotImplementedError("owner implements this: section 3 of the derivation")


def kinetic_energy(state: Array, params: Params) -> Array:
    """Return the kinetic energy ``T``.

    See section 2 of ``docs/derivations/double_pendulum.md``.

    Parameters
    ----------
    state : jax.Array
        ``(theta1, theta2, omega1, omega2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        Scalar (shape ``()``), in mass times length squared per time squared.
    """
    raise NotImplementedError("owner implements this: section 2 of the derivation")


def potential_energy(state: Array, params: Params) -> Array:
    """Return the gravitational potential energy ``V``, zero at the pivot height.

    See section 2 of ``docs/derivations/double_pendulum.md``.

    Parameters
    ----------
    state : jax.Array
        ``(theta1, theta2, omega1, omega2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        Scalar (shape ``()``), in the same units as the kinetic energy.
    """
    raise NotImplementedError("owner implements this: section 2 of the derivation")


def energy(state: Array, params: Params) -> Array:
    """Return the total energy ``E = T + V``.

    Parameters
    ----------
    state : jax.Array
        ``(theta1, theta2, omega1, omega2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        Scalar (shape ``()``), in the same units as the kinetic energy.
    """
    raise NotImplementedError("owner implements this: kinetic plus potential energy")


def acceleration(state: Array, params: Params) -> Array:
    """Return the angular accelerations ``(alpha1, alpha2)`` that solve ``M alpha = b``.

    See section 3 of ``docs/derivations/double_pendulum.md``. Either a closed form or
    ``jnp.linalg.solve`` is acceptable (D-033); the result must also work under
    ``jax.jacfwd``.

    Parameters
    ----------
    state : jax.Array
        ``(theta1, theta2, omega1, omega2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        ``(alpha1, alpha2)``, shape ``(2,)``, in rad per unit time squared.
    """
    raise NotImplementedError("owner implements this: section 3 of the derivation")


def vector_field(state: Array, params: Params) -> Array:
    """Return the time derivative of the state, ``(omega1, omega2, alpha1, alpha2)``.

    The system is autonomous, so there is no time argument; the SciPy wrapper added in
    milestone 3 adapts this function to ``solve_ivp``'s ``f(t, y)``.

    Parameters
    ----------
    state : jax.Array
        ``(theta1, theta2, omega1, omega2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        ``d(state)/dt``, shape ``(4,)``.
    """
    raise NotImplementedError("owner implements this: velocities followed by accelerations")
