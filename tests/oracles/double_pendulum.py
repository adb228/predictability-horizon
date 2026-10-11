"""Independent oracle for the double pendulum (D-029).

Everything here is derived from the geometry of two point masses on rigid rods, with SymPy:
Cartesian positions give the kinetic and potential energy, SymPy forms the Euler-Lagrange
equations and solves for the accelerations. No equation of motion is typed by hand in the
derivation. Numbers are evaluated at ``ORACLE_DIGITS`` digits with mpmath and rounded to
float64 only at the end.

The oracle uses NumPy, SymPy and mpmath only, never JAX, so it shares no code with the
system under test. It also shares no *conventions* with it by accident: the brief's energy
formulas and the derivation note's equations appear below only as **claims** (typed from
those documents) that the self-checks in ``tests/test_oracle_double_pendulum.py`` compare
against the independent derivation.

Conventions: state ``(theta1, theta2, omega1, omega2)``, both angles absolute from the
downward vertical; parameters ``(m1, m2, L1, L2, g)`` in that order.

Why not the float64-lambdified SymPy expression? The simplified closed form has cancellations
and measured about 1e-12 relative error (D-029 evidence), worse than a plain float64 solve,
so the authoritative values are evaluated at high precision instead.
"""

import functools
from collections.abc import Sequence
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import sympy as sp

from horizon.tolerances import ORACLE_DIGITS

STATE_SIZE = 4
N_PARAMS = 5


def to_mp(x: float) -> mp.mpf:
    """Convert a float64 to an mpmath number exactly.

    This is the only conversion the oracle uses. A float64 has a 53-bit mantissa and the
    working precision is far larger, so ``mpf(float(x))`` is exact (tested as O6).
    """
    return mp.mpf(float(x))


@functools.cache
def model() -> SimpleNamespace:
    """Derive the oracle once and return its symbolic objects and mpmath evaluators.

    Returns
    -------
    types.SimpleNamespace
        Symbols ``q1, q2, v1, v2`` (state), ``A1, A2`` (accelerations), ``P`` (the tuple
        ``(m1, m2, L1, L2, g)``); the independent derivation (``T_cart``, ``V_cart``,
        ``M_oracle``, ``b_oracle``, ``alpha``, ``J_sym``); the claims typed from the brief
        and the derivation note (``T_brief``, ``V_brief``, ``M_doc``, ``b_doc``, ``det_doc``,
        ``alpha_doc``); and the mpmath evaluators used by the module-level functions.
    """
    t = sp.Symbol("t")
    m1, m2, L1, L2, g = params = sp.symbols("m1 m2 L1 L2 g", positive=True)
    theta1, theta2 = sp.Function("theta1")(t), sp.Function("theta2")(t)
    omega1, omega2 = theta1.diff(t), theta2.diff(t)
    accel1, accel2 = theta1.diff(t, 2), theta2.diff(t, 2)

    # ---- Independent derivation ------------------------------------------------------------
    # Cartesian positions, y pointing up, angles absolute from the downward vertical.
    x1, y1 = L1 * sp.sin(theta1), -L1 * sp.cos(theta1)
    x2, y2 = x1 + L2 * sp.sin(theta2), y1 - L2 * sp.cos(theta2)
    half = sp.Rational(1, 2)
    kinetic = half * m1 * (x1.diff(t) ** 2 + y1.diff(t) ** 2)
    kinetic += half * m2 * (x2.diff(t) ** 2 + y2.diff(t) ** 2)
    potential = m1 * g * y1 + m2 * g * y2
    lagrangian = kinetic - potential
    euler_lagrange = [
        lagrangian.diff(w).diff(t) - lagrangian.diff(th)
        for w, th in ((omega1, theta1), (omega2, theta2))
    ]
    solution = sp.solve(euler_lagrange, [accel1, accel2], dict=True)[0]

    # Plain symbols in place of functions of time (highest derivatives first).
    q1, q2, v1, v2, A1, A2 = sp.symbols("q1 q2 v1 v2 A1 A2")
    to_symbols = [
        (accel1, A1),
        (accel2, A2),
        (omega1, v1),
        (omega2, v2),
        (theta1, q1),
        (theta2, q2),
    ]
    state = (q1, q2, v1, v2)
    T_cart = kinetic.subs(to_symbols)
    V_cart = potential.subs(to_symbols)
    residual_raw = [e.subs(to_symbols) for e in euler_lagrange]
    # Simplify first so sin^2 + cos^2 pairs collapse; the monomials of the result are the
    # natural terms of M alpha - b. Then expand with trig=False, stated explicitly so the
    # monomial set does not depend on SymPy defaults (D-029).
    residual = [sp.simplify(e) for e in residual_raw]
    monomials = [list(sp.Add.make_args(sp.expand(e, trig=False))) for e in residual]
    M_oracle = sp.Matrix([[e.diff(A) for A in (A1, A2)] for e in residual_raw])
    b_oracle = -sp.Matrix([e.subs({A1: 0, A2: 0}) for e in residual_raw])
    alpha = [sp.simplify(solution[a].subs(to_symbols)) for a in (accel1, accel2)]
    J_sym = sp.Matrix([v1, v2, alpha[0], alpha[1]]).jacobian(sp.Matrix(state))

    # ---- Claims under test: typed from the brief and from docs/derivations ----------------
    cos_d, sin_d = sp.cos(q1 - q2), sp.sin(q1 - q2)
    T_brief = (
        half * (m1 + m2) * L1**2 * v1**2
        + half * m2 * L2**2 * v2**2
        + m2 * L1 * L2 * v1 * v2 * cos_d
    )
    V_brief = -(m1 + m2) * g * L1 * sp.cos(q1) - m2 * g * L2 * sp.cos(q2)
    M_doc = sp.Matrix(
        [[(m1 + m2) * L1**2, m2 * L1 * L2 * cos_d], [m2 * L1 * L2 * cos_d, m2 * L2**2]]
    )
    b_doc = sp.Matrix(
        [
            -m2 * L1 * L2 * sin_d * v2**2 - (m1 + m2) * g * L1 * sp.sin(q1),
            m2 * L1 * L2 * sin_d * v1**2 - m2 * g * L2 * sp.sin(q2),
        ]
    )
    det_doc = L1**2 * L2**2 * m2 * (m1 + m2 * sin_d**2)
    alpha_doc = [
        (M_doc[1, 1] * b_doc[0] - M_doc[0, 1] * b_doc[1]) / det_doc,
        (M_doc[0, 0] * b_doc[1] - M_doc[0, 1] * b_doc[0]) / det_doc,
    ]

    # ---- mpmath evaluators ----------------------------------------------------------------
    def evaluator(args: tuple, expr: object) -> object:
        return sp.lambdify(args, expr, modules="mpmath")

    with_params = state + params
    with_accel = state + (A1, A2) + params
    return SimpleNamespace(
        q1=q1,
        q2=q2,
        v1=v1,
        v2=v2,
        A1=A1,
        A2=A2,
        P=params,
        T_cart=T_cart,
        V_cart=V_cart,
        M_oracle=M_oracle,
        b_oracle=b_oracle,
        alpha=alpha,
        J_sym=J_sym,
        T_brief=T_brief,
        V_brief=V_brief,
        M_doc=M_doc,
        b_doc=b_doc,
        det_doc=det_doc,
        alpha_doc=alpha_doc,
        eval_alpha=evaluator(with_params, alpha),
        eval_mass=evaluator(with_params, M_oracle),
        eval_kinetic=evaluator(with_params, T_cart),
        eval_potential=evaluator(with_params, V_cart),
        eval_jacobian=evaluator(with_params, J_sym),
        eval_residual=evaluator(with_accel, residual),
        eval_terms=[evaluator(with_accel, terms) for terms in monomials],
    )


def _mp_args(state: Sequence[float], params: Sequence[float]) -> tuple[list, list]:
    s = np.asarray(state, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    if s.shape != (STATE_SIZE,):
        raise ValueError(f"state must have shape ({STATE_SIZE},), got {s.shape}")
    if p.shape != (N_PARAMS,):
        raise ValueError(f"params must have shape ({N_PARAMS},), got {p.shape}")
    return [to_mp(v) for v in s], [to_mp(v) for v in p]


def acceleration(state: Sequence[float], params: Sequence[float]) -> np.ndarray:
    """Return the angular accelerations ``(alpha1, alpha2)``, rounded to float64, shape (2,)."""
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args(state, params)
        a = model().eval_alpha(*s, *p)
        return np.array([float(a[0]), float(a[1])])


def mass_matrix(theta: Sequence[float], params: Sequence[float]) -> np.ndarray:
    """Return the mass matrix ``M(theta)``, rounded to float64, shape (2, 2)."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (2,):
        raise ValueError(f"theta must have shape (2,), got {theta.shape}")
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args([*theta, 0.0, 0.0], params)
        return np.array(model().eval_mass(*s, *p).tolist(), dtype=np.float64)


def kinetic_energy(state: Sequence[float], params: Sequence[float]) -> float:
    """Return ``T`` from the Cartesian derivation, rounded to float64."""
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args(state, params)
        return float(model().eval_kinetic(*s, *p))


def potential_energy(state: Sequence[float], params: Sequence[float]) -> float:
    """Return ``V`` from the Cartesian derivation, rounded to float64."""
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args(state, params)
        return float(model().eval_potential(*s, *p))


def energy(state: Sequence[float], params: Sequence[float]) -> float:
    """Return ``T + V`` summed at high precision, then rounded to float64."""
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args(state, params)
        m = model()
        return float(m.eval_kinetic(*s, *p) + m.eval_potential(*s, *p))


def jacobian(state: Sequence[float], params: Sequence[float]) -> np.ndarray:
    """Return the Jacobian of ``(omega1, omega2, alpha1, alpha2)`` w.r.t. the state, (4, 4)."""
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args(state, params)
        return np.array(model().eval_jacobian(*s, *p).tolist(), dtype=np.float64)


def el_residual(
    state: Sequence[float], alpha: Sequence[float], params: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the Euler-Lagrange residual of a *given* acceleration at high precision.

    The float64 ``alpha`` is converted exactly, so the returned residual measures exactly how
    far that float64 acceleration is from satisfying the equations of motion.

    Parameters
    ----------
    state : sequence of float
        ``(theta1, theta2, omega1, omega2)``.
    alpha : sequence of float
        Accelerations ``(alpha1, alpha2)`` to test, shape (2,).
    params : sequence of float
        ``(m1, m2, L1, L2, g)``.

    Returns
    -------
    residual : numpy.ndarray
        Residual of each Euler-Lagrange component, shape (2,).
    scale : numpy.ndarray
        Natural scale of each component, shape (2,): the sum of the absolute values of the
        monomials of the simplified, ``expand(trig=False)``-expanded residual (D-029).
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != (2,):
        raise ValueError(f"alpha must have shape (2,), got {alpha.shape}")
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args(state, params)
        a = [to_mp(v) for v in alpha]
        return _residual_and_scale(s, a, p)


def self_check_residual(
    state: Sequence[float], params: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Residual of the oracle's own, *unrounded* 60-digit accelerations (O2).

    This substitutes the oracle's solution back into its own equations, so it checks only
    the solve step; it is not an independent check (D-029).
    """
    with mp.workdps(ORACLE_DIGITS):
        s, p = _mp_args(state, params)
        a = model().eval_alpha(*s, *p)
        return _residual_and_scale(s, list(a), p)


def _residual_and_scale(s: list, a: list, p: list) -> tuple[np.ndarray, np.ndarray]:
    m = model()
    residual = m.eval_residual(*s, *a, *p)
    scale = [sum(abs(term) for term in terms(*s, *a, *p)) for terms in m.eval_terms]
    return (
        np.array([float(r) for r in residual], dtype=np.float64),
        np.array([float(v) for v in scale], dtype=np.float64),
    )
