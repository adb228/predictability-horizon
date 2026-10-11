"""Predeclared tolerances and the single comparison rule behind every numeric gate.

Governed by D-025 and D-031 in ``docs/DECISIONS.md``. Every value below was recorded in that
log before any test ran against the code it governs. **Changing any value here requires
Opus 5.5's approval first, logged in the decision log (D-025).** The git history of this
file is the audit trail.

The rule
--------
A comparison passes when, elementwise,

    |x - x_ref| <= ATOL + RTOL * s

where ``s`` is the *scale* of the comparison:

* value comparisons: ``s = max(||x_ref||_inf, 1.0)``, taken per case over the whole compared
  array. All quantities are nondimensional and of order one or larger over the case ranges.
* residual checks: ``s`` is the sum of the absolute values of the individual terms that
  cancel in the residual, so rounding error is judged against the size of what was cancelled.

``check_close`` implements the rule once; use it everywhere instead of writing a new
comparison. It fails on NaN or inf, requires exactly equal shapes (no broadcasting), and
rejects complex input unless the caller passes it explicitly.

Tolerance classes
-----------------
* ``ORACLE_SELF_CHECK``: high-precision self-checks of the oracle.
* ``IMPL``: anything compared with the high-precision oracle.
* ``IDENTITY``: exact identities, where only a few ulp of rounding are expected.

Reproducibility of the test cases is tied to the NumPy version locked in ``uv.lock``.
"""

import math

import numpy as np
from numpy.typing import ArrayLike

# --- Oracle precision and self-checks -------------------------------------------------------

ORACLE_DIGITS: int = 60
"""Working precision of the oracle in decimal digits. A setting, not a tolerance.

Basis: float64 carries about 16 digits, so 60 digits leaves a wide margin for cancellation in
the evaluated expressions (the oracle's own residuals come out near 1e-59).
"""

ORACLE_SELF_CHECK_TOL: float = 1e-40
"""Used as both ``atol`` and ``rtol`` for the oracle's 60-digit self-checks.

Basis: observed residuals were about 1e-59 absolute; a real error in the oracle would be of
order one, so the gap between the two is more than 1e30.
"""

# --- Comparing the owner's float64 code with the oracle -------------------------------------

IMPL_RTOL: float = 1e-10
"""Relative tolerance for anything compared with the oracle.

Basis: an honest float64 implementation measured about 1e-13 relative, so this keeps a margin
of 1e3 or more, while structural bugs (a sign, a swapped parameter, the wrong coupling) give
errors of order 1e-3 to 1 at generic states.
"""

IMPL_ATOL: float = 1e-12
"""Absolute tolerance for anything compared with the oracle. Same basis as ``IMPL_RTOL``."""

# --- Exact identities -----------------------------------------------------------------------

IDENTITY_RTOL: float = 1e-12
"""Relative tolerance for exact identities (symmetries, periodicity, scaling, stationarity).

Basis: a few ulp times ``|theta|`` for the moderate angles used (at most about 19 radians).
"""

IDENTITY_ATOL: float = 1e-12
"""Absolute tolerance for exact identities. Same basis as ``IDENTITY_RTOL``."""

# --- Case generation (frozen before any case was generated) ---------------------------------

CASES_SEED: int = 20261010
"""Seed for the case streams. Fixed before any frozen case was generated; never changed.

Parameter set ``k`` uses ``np.random.default_rng([CASES_SEED, k])``.
"""

N_PARAM_SETS: int = 20
"""Number of seeded non-default parameter sets."""

N_STATES: int = 25
"""Number of random states drawn for each parameter set."""

PARAM_LOW: float = 0.5
PARAM_HIGH: float = 2.0
"""Each of ``m1, m2, L1, L2, g`` is log-uniform in ``[PARAM_LOW, PARAM_HIGH]``."""

MIN_PARAM_SEPARATION: float = 0.10
"""Every pair of the five parameters must differ by at least this fraction.

``|a - b| / max(a, b) >= MIN_PARAM_SEPARATION``, so a swap among ``m1, m2, L1, L2, g`` changes
the output and cannot hide behind equal values.
"""

THETA_MAX: float = 4.0 * math.pi
"""Angles are uniform in ``[-THETA_MAX, THETA_MAX]``, deliberately beyond ``2 pi``."""

OMEGA_MAX: float = 5.0
"""Angular velocities are uniform in ``[-OMEGA_MAX, OMEGA_MAX]``."""

N_LARGE_ANGLE_STATES: int = 5
"""Large-angle states (D-032), using parameter sets ``0`` to ``N_LARGE_ANGLE_STATES - 1``."""

LARGE_THETA_MAX: float = 1e3
"""Large-angle states have angles uniform in ``[-LARGE_THETA_MAX, LARGE_THETA_MAX]``."""

LARGE_ANGLE_STREAM: int = 1000
"""Large-angle states come from ``np.random.default_rng([CASES_SEED, LARGE_ANGLE_STREAM])``."""

SCALE_FACTORS: tuple[float, float] = (0.5, 3.0)
"""Factors used by the scaling-invariance tests."""

PERIODICITY_SHIFTS: tuple[int, int] = (-1, 1)
"""Integer multiples of ``2 pi`` added to each angle in the periodicity test."""


# --- The comparison rule --------------------------------------------------------------------


def check_close(
    x: ArrayLike,
    x_ref: ArrayLike,
    *,
    rtol: float,
    atol: float,
    scale: ArrayLike | None = None,
    allow_complex: bool = False,
    what: str = "value",
) -> float:
    """Check ``|x - x_ref| <= atol + rtol * s`` elementwise and raise if it fails.

    Parameters
    ----------
    x : array_like
        Computed value.
    x_ref : array_like
        Reference value. Must have exactly the same shape as ``x``; nothing is broadcast.
    rtol : float
        Relative tolerance, finite and non-negative. Use a class from this module.
    atol : float
        Absolute tolerance, finite and non-negative. Use a class from this module.
    scale : array_like, optional
        The scale ``s``: a scalar, or an array of exactly the shape of ``x``. If omitted,
        ``s = max(||x_ref||_inf, 1.0)`` over the whole array (value comparisons).
    allow_complex : bool, default False
        Complex input is rejected unless this is True.
    what : str, default "value"
        Name used in the failure message.

    Returns
    -------
    float
        The largest ratio ``|x - x_ref| / (atol + rtol * s)`` over all elements. A passing
        check returns a value of at most 1; the closer to 1, the thinner the margin.

    Raises
    ------
    AssertionError
        If any element violates the rule, including any NaN or inf, or if the shapes differ.
    TypeError
        If either input is complex and ``allow_complex`` is False.
    ValueError
        If a tolerance or scale is negative or not finite, or if the arrays are empty (an
        empty comparison would pass vacuously).
    """
    for name, tol in (("rtol", rtol), ("atol", atol)):
        if not (math.isfinite(tol) and tol >= 0.0):
            raise ValueError(f"{name} must be finite and non-negative, got {tol!r}")
    a = np.asarray(x)
    b = np.asarray(x_ref)
    if not allow_complex and (np.iscomplexobj(a) or np.iscomplexobj(b)):
        raise TypeError(f"{what}: complex input needs allow_complex=True")
    if a.shape != b.shape:
        raise AssertionError(f"{what}: shape mismatch, {a.shape} versus {b.shape}")
    if a.size == 0:
        raise ValueError(f"{what}: refusing an empty comparison, which would pass vacuously")
    # NaN and inf must fail outright. Testing only ``err <= allowed`` is not enough: with an
    # infinite reference both sides can be inf, and ``inf <= inf`` is True.
    non_finite = ~(np.isfinite(a) & np.isfinite(b))
    if np.any(non_finite):
        first = tuple(int(i) for i in np.argwhere(non_finite)[0])
        raise AssertionError(
            f"{what}: {int(np.sum(non_finite))} of {a.size} entries are NaN or inf; "
            f"first at index {first}: x={a[first]!r}, x_ref={b[first]!r}"
        )

    if scale is None:
        s = np.full(b.shape, max(float(np.max(np.abs(b))), 1.0))
    else:
        s = np.asarray(scale, dtype=float)
        if s.shape not in ((), b.shape):
            raise ValueError(f"{what}: scale must be a scalar or shape {b.shape}, got {s.shape}")
        s = np.broadcast_to(s, b.shape)
        if not (np.all(np.isfinite(s)) and np.all(s >= 0.0)):
            raise ValueError(f"{what}: scale must be finite and non-negative")

    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs(a - b)
        allowed = atol + rtol * s
        ratio = np.where(allowed > 0.0, err / np.where(allowed > 0.0, allowed, 1.0), np.inf)
        ratio = np.where(err == 0.0, 0.0, ratio)
    if not np.all(err <= allowed):  # written so that a NaN fails instead of slipping through
        score = np.where(np.isnan(ratio), np.inf, ratio)
        worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(score)), score.shape))
        n_bad = int(np.sum(~(err <= allowed)))
        raise AssertionError(
            f"{what}: {n_bad} of {a.size} entries violate |x - x_ref| <= atol + rtol * s "
            f"(atol={atol:g}, rtol={rtol:g}). Worst at index {worst}: x={a[worst]!r}, "
            f"x_ref={b[worst]!r}, |x - x_ref|={err[worst]:.3e}, allowed={allowed[worst]:.3e}, "
            f"ratio={score[worst]:.3g}"
        )
    return float(np.max(ratio))
