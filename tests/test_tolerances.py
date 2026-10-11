"""Tests of the comparison rule and a tripwire for the frozen tolerance register (D-025, D-031).

``check_close`` is the single comparison behind every numeric gate, so it must be able to
fail: each failure mode that matters has a test here.
"""

import math

import numpy as np
import pytest

from horizon import tolerances as tol
from horizon.tolerances import check_close

RTOL, ATOL = 1e-10, 1e-12


def test_passes_inside_the_tolerance_and_reports_the_margin_used() -> None:
    # s = max(|x_ref|, 1) = 2, so allowed = 1e-12 + 1e-10 * 2. Take the error from the floats
    # themselves: 2 + 1e-13 is not exactly representable.
    x_second = 2.0 + 1e-13
    ratio = check_close([1.0, x_second], [1.0, 2.0], rtol=RTOL, atol=ATOL)
    assert 0.0 < ratio < 1.0
    assert ratio == pytest.approx((x_second - 2.0) / (ATOL + RTOL * 2.0))


def test_identical_values_have_zero_ratio() -> None:
    assert check_close([1.5, -2.5], [1.5, -2.5], rtol=RTOL, atol=ATOL) == 0.0


def test_fails_outside_the_tolerance_and_names_the_worst_offender() -> None:
    with pytest.raises(AssertionError, match=r"index \(1,\)") as info:
        check_close([1.0, 2.0 + 1e-6], [1.0, 2.0], rtol=RTOL, atol=ATOL, what="acceleration")
    message = str(info.value)
    assert "acceleration" in message
    assert "ratio=" in message
    assert "1 of 2 entries" in message


def test_the_boundary_is_inclusive_and_one_ulp_beyond_fails() -> None:
    # x_ref = 2 gives s = 2, so allowed = 0.5 * 2 = 1 exactly, and 3 - 2 = 1 exactly.
    assert check_close(3.0, 2.0, rtol=0.5, atol=0.0) == 1.0
    with pytest.raises(AssertionError):
        check_close(np.nextafter(3.0, np.inf), 2.0, rtol=0.5, atol=0.0)


def test_default_scale_has_a_floor_of_one_and_grows_with_the_reference() -> None:
    # With a zero reference the floor makes s = 1, so allowed = atol + rtol = 2e-12. An error of
    # 1.5e-12 sits between atol and atol + rtol, so it passes only if the floor is applied.
    check_close(1.5e-12, 0.0, rtol=1e-12, atol=1e-12)
    with pytest.raises(AssertionError):
        check_close(3e-12, 0.0, rtol=1e-12, atol=1e-12)
    # The same absolute error of 5e-5 is fine against a reference of size 1e6, not of size 1.
    check_close(1e6 + 5e-5, 1e6, rtol=RTOL, atol=ATOL)
    with pytest.raises(AssertionError):
        check_close(1.0 + 5e-5, 1.0, rtol=RTOL, atol=ATOL)


def test_an_explicit_scale_applies_per_component() -> None:
    scale = [1e6, 1.0]  # a residual whose first component cancels terms of size 1e6
    check_close([5e-5, 0.0], [0.0, 0.0], rtol=RTOL, atol=ATOL, scale=scale)
    with pytest.raises(AssertionError, match=r"index \(1,\)"):
        check_close([0.0, 5e-5], [0.0, 0.0], rtol=RTOL, atol=ATOL, scale=scale)
    check_close([5e-5, 0.0], [0.0, 0.0], rtol=RTOL, atol=ATOL, scale=1e6 * np.ones(2))


@pytest.mark.parametrize(
    ("x", "x_ref"),
    [
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, np.nan]),
        ([1.0, np.inf], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, np.inf]),  # inf <= inf is True, so this needs an explicit check
        ([1.0, np.inf], [1.0, np.inf]),
        ([1.0, -np.inf], [1.0, np.inf]),
    ],
    ids=["nan-x", "nan-ref", "inf-x", "inf-ref", "inf-both", "inf-opposite"],
)
def test_any_nan_or_inf_fails(x: list[float], x_ref: list[float]) -> None:
    with pytest.raises(AssertionError, match="NaN or inf"):
        check_close(x, x_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("x_shape", "ref_shape"),
    [((2,), (1, 2)), ((), (2,)), ((2, 1), (1, 2)), ((3,), (2,))],
    ids=["vector-vs-row", "scalar-vs-vector", "column-vs-row", "length"],
)
def test_shapes_must_match_exactly_without_broadcasting(
    x_shape: tuple[int, ...], ref_shape: tuple[int, ...]
) -> None:
    with pytest.raises(AssertionError, match="shape mismatch"):
        check_close(np.zeros(x_shape), np.zeros(ref_shape), rtol=RTOL, atol=ATOL)


def test_complex_input_is_rejected_unless_requested() -> None:
    with pytest.raises(TypeError, match="allow_complex"):
        check_close(1.0 + 1e-15j, 1.0, rtol=RTOL, atol=ATOL)
    with pytest.raises(TypeError, match="allow_complex"):
        check_close(1.0, 1.0 + 0j, rtol=RTOL, atol=ATOL)
    check_close(1.0 + 1e-15j, 1.0 + 0j, rtol=RTOL, atol=ATOL, allow_complex=True)
    with pytest.raises(AssertionError):
        check_close(1.0 + 1e-3j, 1.0 + 0j, rtol=RTOL, atol=ATOL, allow_complex=True)


def test_an_empty_comparison_is_refused_because_it_would_pass_vacuously() -> None:
    with pytest.raises(ValueError, match="empty"):
        check_close(np.zeros(0), np.zeros(0), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("bad", [-1e-12, math.nan, math.inf])
def test_tolerances_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="rtol"):
        check_close(1.0, 1.0, rtol=bad, atol=ATOL)
    with pytest.raises(ValueError, match="atol"):
        check_close(1.0, 1.0, rtol=RTOL, atol=bad)


@pytest.mark.parametrize(
    "scale",
    [-1.0, math.nan, math.inf, np.ones(3)],
    ids=["negative", "nan", "inf", "wrong-shape"],
)
def test_an_invalid_scale_is_rejected(scale: object) -> None:
    with pytest.raises(ValueError, match="scale"):
        check_close([1.0, 2.0], [1.0, 2.0], rtol=RTOL, atol=ATOL, scale=scale)


def test_register_holds_the_frozen_values() -> None:
    """Tripwire: these literals are the values logged in D-031 before any test ran.

    A change to ``src/horizon/tolerances.py`` must also change this test, so the edit shows up
    in two places in the diff. Changing a value needs Opus 5.5's approval first (D-025).
    """
    assert tol.ORACLE_DIGITS == 60
    assert tol.ORACLE_SELF_CHECK_TOL == 1e-40
    assert (tol.IMPL_RTOL, tol.IMPL_ATOL) == (1e-10, 1e-12)
    assert (tol.IDENTITY_RTOL, tol.IDENTITY_ATOL) == (1e-12, 1e-12)
    assert tol.CASES_SEED == 20261010
    assert (tol.N_PARAM_SETS, tol.N_STATES) == (20, 25)
    assert (tol.PARAM_LOW, tol.PARAM_HIGH, tol.MIN_PARAM_SEPARATION) == (0.5, 2.0, 0.10)
    assert 4.0 * math.pi == tol.THETA_MAX
    assert tol.OMEGA_MAX == 5.0
    assert (tol.N_LARGE_ANGLE_STATES, tol.LARGE_THETA_MAX, tol.LARGE_ANGLE_STREAM) == (5, 1e3, 1000)
    assert tol.SCALE_FACTORS == (0.5, 3.0)
    assert tol.PERIODICITY_SHIFTS == (-1, 1)
