"""Self-checks of the independent oracle (O1 to O7, D-029, D-035).

These checks do not call the mechanics implementation. They check the *oracle*, not the code
under test. Which of them are independent anchors, and which are not:

* O1 (energies equal the brief's), O3 (the derivation note equals the oracle) and O4 (normal
  modes are exactly 2 +/- sqrt(2)) are independent anchors.
* O2 substitutes the oracle's own solution back into its own equations, so it is close to
  circular and only checks the solve step.
* O5, O6 and O7 guard the test machinery (the cases, the float-to-high-precision conversion,
  and the absence of symbolic computation when the implementation is tested).
"""

import ast
import itertools
import os
import pathlib
import sys
from collections.abc import Callable
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from horizon import tolerances as tol
from horizon.tolerances import check_close
from tests import cases
from tests.oracles import double_pendulum as oracle

MODERATE = cases.moderate_cases()
LARGE_ANGLE = cases.large_angle_cases()

# The only oracle functions the tests of the implementation may call (O7).
NUMERIC_API = frozenset(
    {
        "mass_matrix",
        "kinetic_energy",
        "potential_energy",
        "energy",
        "acceleration",
        "jacobian",
        "el_residual",
    }
)


def test_o1_cartesian_energies_equal_the_briefs_symbolically() -> None:
    """The brief's T and V equal the energies derived from Cartesian geometry.

    Independent anchor: it verifies the brief's formulas, including the convention that both
    angles are absolute from the downward vertical, instead of assuming them.
    """
    m = oracle.model()
    assert sp.simplify(m.T_cart - m.T_brief) == 0
    assert sp.simplify(m.V_cart - m.V_brief) == 0


@pytest.mark.parametrize("case", MODERATE, ids=lambda c: f"set-{c.index:02d}")
def test_o2_oracle_solution_satisfies_its_euler_lagrange_equations(case: cases.Case) -> None:
    """The unrounded 60-digit accelerations leave a residual far below ORACLE_SELF_CHECK_TOL.

    Close to circular (it substitutes the oracle's solution into its own equations), so it
    only checks the solve step. Scale: sum of the absolute values of the monomials.
    """
    for state in case.states:
        residual, scale = oracle.self_check_residual(state, case.params)
        check_close(
            residual,
            np.zeros(2),
            rtol=tol.ORACLE_SELF_CHECK_TOL,
            atol=tol.ORACLE_SELF_CHECK_TOL,
            scale=scale,
            what=f"EL residual at {state.tolist()}",
        )


def test_o2_also_holds_at_the_large_angle_states() -> None:
    """The same self-check at the five large-angle states (D-032)."""
    for case in LARGE_ANGLE:
        residual, scale = oracle.self_check_residual(case.states[0], case.params)
        check_close(
            residual,
            np.zeros(2),
            rtol=tol.ORACLE_SELF_CHECK_TOL,
            atol=tol.ORACLE_SELF_CHECK_TOL,
            scale=scale,
            what=f"EL residual, large-angle state {case.index}",
        )


def test_o3_the_derivation_note_equals_the_oracle_symbolically() -> None:
    """M, b, det M and the closed form for alpha in the note equal the oracle's.

    Independent anchor: it verifies the document itself, so the implementation is written from a
    verified equation.
    """
    m = oracle.model()
    assert sp.simplify(m.M_oracle - m.M_doc) == sp.zeros(2, 2)
    assert sp.simplify(m.b_oracle - m.b_doc) == sp.zeros(2, 1)
    assert sp.simplify(m.M_doc.det() - m.det_doc) == 0
    for from_oracle, from_note in zip(m.alpha, m.alpha_doc, strict=True):
        assert sp.simplify(from_oracle - from_note) == 0


def test_o4_oracle_normal_modes_are_exactly_two_plus_minus_sqrt_two() -> None:
    """At the hanging state with default parameters the oracle's Jacobian is exactly as claimed.

    Independent anchor: the normal modes come from the brief, not from either implementation.
    Checks the block structure, the characteristic polynomial of the 4x4 Jacobian (whose roots
    are +/- i sqrt(2 +/- sqrt(2))), and that of the stiffness block M(0)^-1 K (roots 2 +/- sqrt(2)).
    """
    m = oracle.model()
    at_rest = {m.q1: 0, m.q2: 0, m.v1: 0, m.v2: 0}
    defaults = dict.fromkeys(m.P, 1)
    jacobian = m.J_sym.subs(at_rest).subs(defaults)

    assert jacobian[:2, :2] == sp.zeros(2, 2)
    assert jacobian[:2, 2:] == sp.eye(2)
    assert jacobian[2:, 2:] == sp.zeros(2, 2)

    lam = sp.Symbol("lam")
    roots = [s * sp.I * sp.sqrt(2 + r * sp.sqrt(2)) for s in (1, -1) for r in (1, -1)]
    expected = sp.expand(sp.Mul(*[lam - root for root in roots]))
    assert sp.expand(jacobian.charpoly(lam).as_expr() - expected) == 0

    stiffness = -jacobian[2:, :2]
    assert sp.expand(stiffness.charpoly(lam).as_expr() - (lam**2 - 4 * lam + 2)) == 0


def test_o5_cases_are_non_default_distinct_reproducible_and_swap_sensitive() -> None:
    """The frozen cases are what D-031 says they are, and a parameter swap is detectable.

    Beyond the 10 percent separation, this checks the claim that motivates it: swapping any two
    of (m1, m2, L1, L2, g) is rejected by the IMPL tolerance, on every frozen parameter set.
    """
    assert len(MODERATE) == tol.N_PARAM_SETS
    assert len({c.params for c in MODERATE}) == tol.N_PARAM_SETS
    for case in MODERATE:
        assert case.params != (1.0,) * 5, "a set equal to the all-ones defaults hides swap bugs"
        assert cases.min_pairwise_separation(case.params) >= tol.MIN_PARAM_SEPARATION
        assert all(tol.PARAM_LOW <= v <= tol.PARAM_HIGH for v in case.params)
        assert case.states.shape == (tol.N_STATES, 4)
        assert np.all(np.isfinite(case.states))
        assert np.all(np.abs(case.states[:, :2]) <= tol.THETA_MAX)
        assert np.all(np.abs(case.states[:, 2:]) <= tol.OMEGA_MAX)
        assert not case.states.flags.writeable
    for case in LARGE_ANGLE:
        assert np.all(np.abs(case.states[:, :2]) <= tol.LARGE_THETA_MAX)
        assert np.all(np.abs(case.states[:, 2:]) <= tol.OMEGA_MAX)
        assert case.params == MODERATE[case.index].params

    # Reproducible: regenerating from the seed, bypassing the cache, gives identical arrays.
    for fresh, cached in zip(cases.moderate_cases.__wrapped__(), MODERATE, strict=True):
        assert fresh.params == cached.params
        np.testing.assert_array_equal(fresh.states, cached.states)

    # Swap sensitivity: a swap is detectable when the very rule that gates the implementation (the
    # IMPL tolerance) rejects the swapped acceleration. Looks at the first three states of each set.
    names = ("m1", "m2", "L1", "L2", "g")
    for case in MODERATE:
        for i, j in itertools.combinations(range(5), 2):
            swapped = list(case.params)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            detected = False
            for state in case.states[:3]:
                try:
                    check_close(
                        oracle.acceleration(state, swapped),
                        oracle.acceleration(state, case.params),
                        rtol=tol.IMPL_RTOL,
                        atol=tol.IMPL_ATOL,
                    )
                except AssertionError:
                    detected = True
                    break
            assert detected, f"swapping {names[i]} and {names[j]} is invisible on set {case.index}"


def test_o6_float_to_high_precision_conversion_is_exact() -> None:
    """``to_mp`` represents every float64 the oracle receives exactly, and round-trips.

    Every state component and parameter of the frozen cases, plus awkward special values, is
    converted; the high-precision value must equal the float's exact rational value.
    """
    values = {0.0, -0.0, 1.0, np.pi, -np.e, 0.1, 1.0 / 3.0, 1e-300, 1e300}
    values |= {float(np.nextafter(1.0, 2.0)), float(np.nextafter(1.0, 0.0)), 2.0**-52}
    for case in (*MODERATE, *LARGE_ANGLE):
        values.update(float(v) for v in case.states.ravel())
        values.update(case.params)
    assert len(values) > 1000
    with mp.workdps(tol.ORACLE_DIGITS):
        for x in values:
            exact = Fraction(x)
            converted = oracle.to_mp(x)
            assert float(converted) == x
            assert converted * exact.denominator == exact.numerator, f"{x!r} is not exact"


def _sympy_calls(run: Callable[[], object]) -> list[str]:
    """Names of the SymPy functions called while ``run()`` executes, in the main thread only.

    ``sys.setprofile`` applies to the current thread alone. The previous profiler is saved and
    restored in a ``finally`` block, so a failure cannot leave the tracer installed.
    """
    package_dir = os.path.join(os.path.dirname(sp.__file__), "")
    called: list[str] = []

    def profiler(frame, event, arg) -> None:
        if event == "call" and frame.f_code.co_filename.startswith(package_dir):
            called.append(frame.f_code.co_qualname)

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        run()
    finally:
        sys.setprofile(previous)
    return called


def test_o7_the_oracle_numeric_functions_call_no_sympy() -> None:
    """Evaluating the oracle at a state runs no SymPy code, so the implementation's tests run none.

    T4 evaluates the implementation's float64 acceleration in the oracle's cached, lambdified
    residual; it never simplifies, expands or solves anything (D-035). This is the check: after
    the symbolic model is built (outside the traced region), the seven numeric functions that the
    tests of the implementation may call must make no call into the ``sympy`` package.
    """
    oracle.model()  # the one-time symbolic derivation; the tracer is installed afterwards
    state, params = MODERATE[0].states[0], MODERATE[0].params
    alpha = oracle.acceleration(state, params)

    def evaluate() -> None:
        oracle.mass_matrix(state[:2], params)
        oracle.kinetic_energy(state, params)
        oracle.potential_energy(state, params)
        oracle.energy(state, params)
        oracle.acceleration(state, params)
        oracle.jacobian(state, params)
        oracle.el_residual(state, alpha, params)

    called = _sympy_calls(evaluate)
    assert not called, f"{len(called)} calls into sympy, first: {called[:5]}"


def test_o7_the_call_tracer_detects_a_real_sympy_call() -> None:
    """Positive control: a gate that cannot fail is not a gate. Also checks the restore."""
    x = sp.Symbol("x")
    before = sys.getprofile()
    called = _sympy_calls(lambda: sp.simplify(sp.sin(x) ** 2 + sp.cos(x) ** 2))
    assert called, "the tracer saw no SymPy call during a simplify"
    assert sys.getprofile() is before, "the tracer was left installed"


def test_o7_the_implementation_tests_reach_the_oracle_only_through_its_numeric_functions() -> None:
    """``tests/test_double_pendulum.py`` imports no SymPy and calls only ``NUMERIC_API``.

    Together with the call trace above, this means that no test of the implementation can run
    symbolic computation, whichever oracle function it uses. Using another oracle function
    (``model``, for example) fails here until it is added to ``NUMERIC_API`` and traced.
    """
    path = pathlib.Path(__file__).with_name("test_double_pendulum.py")
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "sympy" not in imported
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "oracle"
    }
    assert used, "the tests no longer use the oracle, so this check would be vacuous"
    assert used <= NUMERIC_API, f"not in NUMERIC_API: {sorted(used - NUMERIC_API)}"
