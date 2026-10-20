"""Planar double pendulum: mass matrix, energies and equations of motion.

The interface is fixed by D-028. The equations were derived and verified by the project owner
and the function bodies were implemented by the agent (D-039). They are written out in
``docs/derivations/double_pendulum.md``, and the frozen tests in
``tests/test_double_pendulum.py`` define "correct" (D-029 to D-034).

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

import jax.numpy as jnp
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
    m1, m2, L1, L2, _ = params
    coupling = m2 * L1 * L2 * jnp.cos(theta[0] - theta[1])  # M12 = M21, uses theta1 - theta2
    return jnp.array([[(m1 + m2) * L1**2, coupling], [coupling, m2 * L2**2]])


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
    omega = state[2:]
    return 0.5 * (omega @ mass_matrix(state[:2], params) @ omega)  # T = 1/2 omega^T M omega


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
    m1, m2, L1, L2, g = params
    return -(m1 + m2) * g * L1 * jnp.cos(state[0]) - m2 * g * L2 * jnp.cos(state[1])


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
    return kinetic_energy(state, params) + potential_energy(state, params)


def acceleration(state: Array, params: Params) -> Array:
    """Return the angular accelerations ``(alpha1, alpha2)`` that solve ``M alpha = b``.

    See section 3 of ``docs/derivations/double_pendulum.md``. This is the closed-form solution
    of the 2 x 2 system (D-039), written with ``jnp`` only and no branching on values, so it
    also works under ``jit``, ``vmap`` and ``jax.jacfwd``.

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
    m1, m2, L1, L2, g = params
    theta1, theta2, omega1, omega2 = state[0], state[1], state[2], state[3]
    delta = theta1 - theta2
    M = mass_matrix(state[:2], params)

    # Right-hand side of M alpha = b (section 3). Velocities enter only as omega1^2, omega2^2.
    coupling = m2 * L1 * L2 * jnp.sin(delta)
    b1 = -coupling * omega2**2 - (m1 + m2) * g * L1 * jnp.sin(theta1)
    b2 = coupling * omega1**2 - m2 * g * L2 * jnp.sin(theta2)

    # Closed-form 2 x 2 solve; det M in its factored form, which is positive for m1, m2 > 0.
    det = m2 * L1**2 * L2**2 * (m1 + m2 * jnp.sin(delta) ** 2)
    alpha1 = (M[1, 1] * b1 - M[0, 1] * b2) / det
    alpha2 = (M[0, 0] * b2 - M[0, 1] * b1) / det
    return jnp.stack([alpha1, alpha2])


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
    return jnp.concatenate([state[2:], acceleration(state, params)])


# ---- Canonical coordinates (D-078) ---------------------------------------------------------
#
# ``(q, p) = (theta, M(theta) omega)``. The angles ``q`` are never wrapped (D-010). The flow in
# these coordinates preserves phase-space volume, so its exponents sum to exactly 0 (D-008).


def _solve_mass(theta: Array, rhs: Array, params: Params) -> Array:
    """Solve ``M(theta) x = rhs`` by the closed-form 2 x 2 solve of ``acceleration``."""
    m1, m2, L1, L2, _ = params
    M = mass_matrix(theta, params)
    det = m2 * L1**2 * L2**2 * (m1 + m2 * jnp.sin(theta[0] - theta[1]) ** 2)
    return jnp.stack(
        [(M[1, 1] * rhs[0] - M[0, 1] * rhs[1]) / det, (M[0, 0] * rhs[1] - M[0, 1] * rhs[0]) / det]
    )


def to_canonical(state: Array, params: Params) -> Array:
    """Return ``(q, p) = (theta, M(theta) omega)``, shape ``(4,)`` (D-078).

    Parameters
    ----------
    state : jax.Array
        ``(theta1, theta2, omega1, omega2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        ``(q1, q2, p1, p2)``, shape ``(4,)``.
    """
    theta, omega = state[:2], state[2:]
    return jnp.concatenate([theta, mass_matrix(theta, params) @ omega])


def from_canonical(z: Array, params: Params) -> Array:
    """Return ``(theta, omega)`` with ``omega = M(q)^-1 p``, the inverse of ``to_canonical``.

    Parameters
    ----------
    z : jax.Array
        ``(q1, q2, p1, p2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        ``(theta1, theta2, omega1, omega2)``, shape ``(4,)``.
    """
    return jnp.concatenate([z[:2], _solve_mass(z[:2], z[2:], params)])


def hamiltonian(z: Array, params: Params) -> Array:
    """Return ``H = 1/2 p^T M(q)^-1 p + V(q)``, a scalar (D-078).

    Parameters
    ----------
    z : jax.Array
        ``(q1, q2, p1, p2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        Scalar (shape ``()``), equal to ``energy(from_canonical(z))``.
    """
    q, p = z[:2], z[2:]
    omega = _solve_mass(q, p, params)
    return 0.5 * (p @ omega) + potential_energy(jnp.concatenate([q, omega]), params)


def canonical_vector_field(z: Array, params: Params) -> Array:
    """Return ``(q', p')`` with ``q' = M^-1 p`` and ``p' = M' omega + M alpha`` (D-078).

    ``omega = M(q)^-1 p``, ``alpha = acceleration((q, omega))`` and ``M'`` is the time
    derivative of ``M(q)`` along ``q' = omega``. This reuses the verified mechanics; the note
    shows that it equals Hamilton's ``-dH/dq``.

    Parameters
    ----------
    z : jax.Array
        ``(q1, q2, p1, p2)``, shape ``(4,)``.
    params : Params
        Physical parameters ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    jax.Array
        ``dz/dt``, shape ``(4,)``.
    """
    m1, m2, L1, L2, _ = params
    q, p = z[:2], z[2:]
    omega = _solve_mass(q, p, params)
    alpha = acceleration(jnp.concatenate([q, omega]), params)
    # Only M12 = M21 = m2 L1 L2 cos(q1 - q2) depends on q, so M' has that entry's derivative.
    coupling_rate = -m2 * L1 * L2 * jnp.sin(q[0] - q[1]) * (omega[0] - omega[1])
    m_dot_omega = coupling_rate * jnp.stack([omega[1], omega[0]])
    return jnp.concatenate([omega, m_dot_omega + mass_matrix(q, params) @ alpha])
