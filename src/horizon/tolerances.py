"""Predeclared tolerances and the single comparison rule behind every numeric gate.

Governed by D-025 and D-031 in ``docs/DECISIONS.md``, for the integrator gates by D-044 to
D-048, and for the physics validation gates by D-052 to D-060. Every value below was recorded in
that log before any test ran against the code it governs. **Changing any value here requires
Opus 5.5's approval first, logged in the decision log (D-025).** The git history of this file is
the audit trail.

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
from typing import NamedTuple

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

# --- Comparing the implementation's float64 results with the oracle -------------------------

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

# --- Milestone 3: integrator gates (D-044 to D-048) -----------------------------------------
#
# Declared in the decision log before any milestone 3 test ran against the implementation.
# They are not measurements: the calibration behind them is uncommitted lab work described in
# the log. Times of the pendulum gates are in units of ``sqrt(L1 / g)``, which is 1 for the
# default parameters. The owner asked that none of these be changed after seeing results from
# the implementation; a change also needs Opus 5.5's approval first (D-025).
#
# The streams below are ``np.random.default_rng([CASES_SEED, stream])``. The lab used other
# seeds, so no calibration sample is a gate sample.


class ExactGate(NamedTuple):
    """Horizon and error ceiling of one exact-solution problem (D-044)."""

    name: str
    horizon: int
    ceiling: float


EXACT_DT_GRID: tuple[float, float, float, float] = (0.04, 0.02, 0.01, 0.005)
"""Step sizes of the exact-solution gate and of the pendulum convergence gate.

Outputs are at the integer times ``t = 0, 1, ..., T``, so ``stride = round(1 / dt)`` steps
separate two outputs (D-044).
"""

EXACT_SLOPE_BAND: tuple[float, float] = (3.9, 4.1)
"""The least-squares slope of ``log(error)`` against ``log(dt)`` must lie in this band for
each exact-solution problem (D-044). Fourth order means a slope of 4."""

EXACT_GATES: tuple[ExactGate, ...] = (
    ExactGate("harmonic", 10, 3e-8),
    ExactGate("linear", 5, 2e-9),
    ExactGate("logistic", 5, 7.5e-12),
    ExactGate("rotation", 5, 1.1e-8),
)
"""Per problem: the horizon ``T`` and the ceiling on RK4's error at ``dt = 0.005`` (D-044).

Basis: D-044 sized each ceiling at 10 to 18 times the error seen in the calibration
(uncommitted lab work, described in the log), so a wrong constant factor fails and rounding
does not.
"""

EXACT_WRAPPER_TOL: float = 1e-10
"""Largest allowed distance between the DOP853 wrapper (default tolerances) and the exact
solution, on all four exact-solution problems (D-044)."""

CONV_STREAM: int = 2000
"""Stream of the regular states of the convergence gate, also used by the regular agreement
gate (D-045). Drawn as ``rng.uniform(-REGULAR_THETA_MAX, REGULAR_THETA_MAX, size=(N, 2))``."""

N_REGULAR_STATES: int = 12
"""Number of regular states (``omega = 0``) in the convergence and regular agreement gates."""

REGULAR_THETA_MAX: float = 0.4
"""Regular states have both angles uniform in ``[-REGULAR_THETA_MAX, REGULAR_THETA_MAX]``.

The amplitude bound is declared in advance because RK4's slope is pre-asymptotic at larger
amplitudes (D-045).
"""

CONV_T: int = 10
"""Horizon of the convergence gate, with outputs at ``t = 0, 1, ..., CONV_T`` (D-045)."""

CONV_DT_GRID: tuple[float, float, float, float] = (0.04, 0.02, 0.01, 0.005)
"""Step sizes of the convergence gate (D-045). D-045 excluded a finer grid because its smallest
step would sit too close to the reference's own error."""

CONV_REFERENCE_TOL: float = 1e-13
"""``rtol`` and ``atol`` of the reference in the convergence gate and in the regular agreement
gate only (D-045, D-046). The wrapper's default stays ``1e-12``."""

CONV_SELF_CHECK_TOL: float = 2.5e-14
"""``rtol`` and ``atol`` of the second reference used to measure the reference's own error.

Basis: just above SciPy's floor of ``100 * eps = 2.22e-14``, below which SciPy raises ``rtol``
silently (the wrapper turns that into an error).
"""

CONV_SLOPE_BAND: tuple[float, float] = (3.8, 4.2)
"""Every regular state's convergence slope must lie in this band (D-045)."""

CONV_FLOOR_RATIO: float = 30.0
"""Validity rule (D-045): RK4's error at the smallest ``dt`` must be at least this multiple of
the reference's own error, or the test fails as inconclusive instead of passing."""

AGREE_DT: float = 0.01
"""RK4 step of the agreement gates, in units of ``sqrt(L1 / g)`` (D-046)."""

CHAOTIC_STREAM: int = 2001
"""Stream of the chaotic-ish states (D-046). Per attempt ``rng.uniform(-CHAOTIC_THETA_MAX,
CHAOTIC_THETA_MAX, size=2)`` with ``omega = 0``, kept only if ``E > 0``, until
``N_CHAOTIC_STATES`` are kept. "Chaotic-ish" is not a classification (milestone 6)."""

N_CHAOTIC_STATES: int = 30
"""Number of chaotic-ish states."""

CHAOTIC_THETA_MAX: float = math.pi
"""Chaotic-ish states have both angles uniform in ``[-pi, pi]``."""

CHAOTIC_T: int = 8
"""Horizon of the chaotic-ish agreement gate, with outputs at ``t = 0, 1, ..., 8`` (D-046).

The proposal was 10; the predeclared 20-sample tail rule of D-046 moved it to 8.
"""

CHAOTIC_REFERENCE_TOL: float = 1e-12
"""``rtol`` and ``atol`` of the reference in the chaotic-ish agreement gate (the brief's)."""

CHAOTIC_TOL: float = 1e-4
"""Worst case over states, output times and components of ``|RK4 - DOP853|`` (D-046)."""

REGULAR_AGREE_T: int = 50
"""Horizon of the regular agreement gate (D-046), with outputs at every integer time."""

REGULAR_AGREE_TOL: float = 1e-6
"""Worst case over the regular states, output times and components (D-046)."""

WIRE_PARAM_STREAM: int = 2002
"""Stream of the wiring check's parameter sets, drawn one after the other by the D-031 recipe."""

WIRE_STATE_STREAM: int = 2003
"""Stream of the wiring check's states, consumed set by set in order (D-047): per attempt
``rng.uniform(-CHAOTIC_THETA_MAX, CHAOTIC_THETA_MAX, size=2)`` with ``omega = 0``, kept only if
``E > 0`` under that set's parameters."""

N_WIRE_SETS: int = 10
"""Number of non-default parameter sets in the wiring check."""

N_WIRE_STATES: int = 3
"""Number of states per parameter set in the wiring check."""

WIRE_DT: float = 0.005
"""RK4 step of the wiring check, in units of ``t_s = sqrt(L1 / g)`` of each set (D-047)."""

WIRE_T: float = 2.0
"""Horizon of the wiring check in units of ``t_s``, with outputs at ``0, t_s, 2 t_s``."""

WIRE_REFERENCE_TOL: float = 1e-12
"""``rtol`` and ``atol`` of the reference in the wiring check."""

WIRE_AGREE_TOL: float = 1e-5
"""Worst-case distance between RK4 and the wrapper, on the nondimensional state
``(theta1, theta2, omega1 * t_s, omega2 * t_s)`` (D-047)."""

WIRE_ENERGY_TOL: float = 1e-5
"""Worst case of ``|E(t) - E(0)| / E_s`` for the RK4 trajectory and for the wrapper's, with ``E``
evaluated under the set's own parameters and ``E_s = g * ((m1 + m2) * L1 + m2 * L2)``, the
largest possible ``|V|`` (D-047)."""


# --- Milestone 4: physics validation (D-052 to D-060) ---------------------------------------
#
# Declared in the decision log before any milestone 4 test ran. They are not measurements: the
# calibration behind them is uncommitted lab work described in the log. Times of the default
# parameter gates are in units of ``sqrt(L1 / g)``, which is 1 for the default parameters. The
# streams are ``np.random.default_rng([CASES_SEED, stream])`` and are disjoint from the earlier
# ones. A change to any value needs Opus 5.5's approval first (D-025). A calibrated ceiling is
# ten times the largest of 20 disjoint calibration sample maxima, rounded up to 1, 2 or 5 times
# a power of ten (D-055); two values are not set that way: ``E_DOP_CEILING`` is the brief's
# number and ``F_TOL`` belongs to deterministic cases.

E_HORIZON: int = 1000
"""Horizon of the energy-drift gates, with outputs at ``t = 0, 1, ..., E_HORIZON`` (D-054)."""

E_DOP_CEILING: float = 1e-9
"""E1: ceiling on ``max_t |E(t) - E(0)| / E_s`` for DOP853 at its defaults. The brief's number,
not derived from data. ``E_s`` is a fixed scale of the parameters alone (D-053, D-054)."""

E1_STREAM_REGULAR: int = 3000
"""Stream of the regular gate states of E1: ``rng.uniform(-REGULAR_THETA_MAX, REGULAR_THETA_MAX,
size=(E1_N_REGULAR, 2))`` with ``omega = 0`` (D-054)."""

E1_N_REGULAR: int = 2
"""Number of regular states in E1."""

E1_STREAM_CHAOTIC: int = 3001
"""Stream of the chaotic-ish gate states of E1, drawn by the recipe of D-046(a) (D-054)."""

E1_N_CHAOTIC: int = 5
"""Number of chaotic-ish states in E1."""

E_TIMEOUT_S: int = 300
"""The ``@pytest.mark.timeout`` of every E1 test, including its controls (D-036, D-054)."""

E_GROWTH_TIMES: tuple[int, int, int] = (100, 300, 1000)
"""Times at which the script prints the running maximum of the drift, the growth law (D-054)."""

E_LITERAL_FLOOR: float = 0.1
"""Report only, never gating: the literal ``|dE| / |E(0)|`` is shown for states with
``|E(0)| >= E_LITERAL_FLOOR * E_s``. ``E`` is ``double_pendulum.energy`` (the milestone 2 zero
of the potential) and ``E_s`` comes from ``energy_scale`` (D-053)."""

E_CONTROL_RTOL: float = 1e-10
"""``rtol`` and ``atol`` of the reference-tolerance control of E1 (D-054, D-058)."""

E_CONTROL_N_STATES: int = 2
"""The control runs on the first ``E_CONTROL_N_STATES`` chaotic-ish gate states of E1."""

E_CONTROL_DAMPING: float = 1e-6
"""Coefficient ``gamma`` of the damping term ``-gamma * omega`` of the controls of E1 and V1
(D-058)."""

E_CONTROL_RK4_DT: float = 0.05
"""Step of RK4 in the control of E2 (D-058)."""

E2_STREAM_REGULAR: int = 3002
"""Stream of the regular states of E2, drawn like those of E1 (D-055)."""

E2_N_REGULAR: int = 12
"""Number of regular states in E2."""

E2_STREAM_CHAOTIC: int = 3003
"""Stream of the chaotic-ish states of E2 (D-055)."""

E2_N_CHAOTIC: int = 30
"""Number of chaotic-ish states in E2."""

E2_CEILING_REGULAR: float = 1e-7
"""E2: ceiling on the RK4 drift (``AGREE_DT``, ``T = E_HORIZON``) over the regular states.
Calibrated, D-055: the largest of 20 sample maxima was 5.44e-9."""

E2_CEILING_CHAOTIC: float = 5e-4
"""E2: ceiling on the RK4 drift over the chaotic-ish states. Calibrated, D-055: the largest of
20 sample maxima was 2.62e-5."""

MOVING_STREAM: int = 3006
"""Stream of the moving states, which the script measures and reports and no test gates: angles
uniform in ``[-CHAOTIC_THETA_MAX, CHAOTIC_THETA_MAX]`` (one draw of shape ``(N_MOVING, 2)``), then
velocities uniform in ``[-MOVING_OMEGA_MAX, MOVING_OMEGA_MAX]`` (a second draw) (D-054)."""

N_MOVING: int = 8
"""Number of moving states."""

MOVING_OMEGA_MAX: float = 2.0
"""Bound of the velocities of the moving states."""

F_AMPLITUDE: float = 1e-3
"""Amplitude ``A`` of the small-angle FFT. The physical amplitude-dependent frequency shift
scales as ``A**2`` and sets the floor under the agreement, so ``A`` is declared and not adjusted
(D-056)."""

F_SAMPLE_DT: float = 0.05
"""Sampling interval of the FFT record, in units of ``t_s = sqrt(L1 / g)`` of the parameter set
(D-056)."""

F_N_SAMPLES: int = 20000
"""Samples in the FFT record, ``t = k * F_SAMPLE_DT * t_s`` for ``k = 0, ..., F_N_SAMPLES - 1``,
so ``T_total = 1000 t_s`` (D-056)."""

F_RK4_STRIDE: int = 5
"""RK4 steps between two samples; the step is ``AGREE_DT * t_s``, so
``F_RK4_STRIDE * AGREE_DT = F_SAMPLE_DT`` (D-056)."""

F_PAD: int = 8
"""The spectrum is zero-padded to the power of two that is at least ``F_PAD * F_N_SAMPLES``
(D-056)."""

F_PEAK_SEPARATION_BINS: int = 40
"""Two peaks of the spectrum must be more than this many padded bins apart (D-056)."""

F_TOL: float = 1e-5
"""Largest allowed relative error of a peak frequency against ``sqrt((g/L) (2 -+ sqrt 2))``,
for both integrators and both parameter sets (D-056). Fixed by deterministic cases, not
calibrated."""

F2_PARAMS: tuple[float, float, float, float, float] = (1.3, 1.3, 0.7, 0.7, 9.81)
"""The second small-angle set, ``(m1, m2, L1, L2, g)``, equal masses and lengths with ``g/L``
different from 1 (D-056)."""

F_CONTROL_M2_FACTOR: float = 1.01
"""``m2`` is multiplied by this in the control of F, against the default analytic frequencies
(D-058)."""

V_REGULAR_T: int = 100
"""Horizon of the round trip of the regular states (D-057)."""

V_CHAOTIC_T: int = 8
"""Horizon of the round trip of the chaotic-ish states, the horizon of milestone 3's agreement
gate (D-057)."""

V_STREAM_REGULAR: int = 3004
"""Stream of the regular states of V1 and V2 (D-057)."""

V_N_REGULAR: int = 12
"""Number of regular states in V1 and V2."""

V_STREAM_CHAOTIC: int = 3005
"""Stream of the chaotic-ish states of V1 and V2 (D-057)."""

V_N_CHAOTIC: int = 30
"""Number of chaotic-ish states in V1 and V2."""

V1_CEILING_REGULAR: float = 5e-9
"""V1: ceiling on the DOP853 round-trip error over the regular states. Calibrated, D-057: the
largest of 20 sample maxima was 4.09e-10."""

V1_CEILING_CHAOTIC: float = 5e-8
"""V1: ceiling over the chaotic-ish states. Calibrated, D-057: the largest of 20 sample maxima
was 4.24e-9."""

V2_CEILING_REGULAR: float = 1e-6
"""V2: ceiling on the RK4 (``AGREE_DT``) round-trip error over the regular states. Calibrated,
D-057: the largest of 20 sample maxima was 9.42e-8."""

V2_CEILING_CHAOTIC: float = 2e-3
"""V2: ceiling over the chaotic-ish states. Calibrated, D-057: the largest of 20 sample maxima
was 1.39e-4. A backstop for gross errors: second-order schemes sit near this ceiling, so it
may or may not see them (D-063)."""

V1_REGULAR_TIMEOUT_S: int = 120
"""The ``@pytest.mark.timeout`` of the V1 test on regular states (D-036, D-063). On the Linux
runner that test took 38.2 s of the default 60 s, a margin of 1.6 times; 120 s is 3.1 times the
runner time. It changes no threshold and no gate value."""


# --- Milestone 5: Lyapunov estimators (D-076 to D-086) ---------------------------------------
#
# Declared in the decision log before any milestone 5 test ran, and frozen with D-086. None of
# these is a measurement: the references are exact invariants, closed forms or the literature,
# and the tolerances come from theory (owner's instruction 1 of the milestone 5 design). The
# calibration lab may set only the pendulum Benettin interval, the QR cadences and the Lorenz
# transient (D-083); those values are added here by a later entry, before any gate test runs.
# The constants of the lab's rules are rule constants, not gate values: ``ln 10`` (the Benettin
# interval), ``10**3`` (the QR cadence), the ladder ``(10, 30, 100, 300, 1000)`` with the
# half-tolerance criterion (the Lorenz transient), and the first lab stream element ``4545``.
# The streams are ``np.random.default_rng([CASES_SEED, stream])``, disjoint from the earlier
# ones. A frame is the ``Q`` of ``np.linalg.qr(rng.standard_normal((n, n)))`` with each column's
# sign set so that ``diag(R) > 0`` (D-084). A change to any value needs Opus 5.5's approval
# first (D-025).

L4_FRAME_STREAM: int = 4000
"""Stream of the L4 frame (D-080, D-084)."""

HO_FRAME_STREAM: int = 4001
"""Stream of the HO frame (D-080, D-084)."""

LORENZ_IC_STREAM: int = 4002
"""Stream of the Lorenz initial conditions (D-081): one draw ``rng.uniform(low, high,
size=(N, 3))`` with ``low = (xy_lo, xy_lo, z_lo)`` and ``high = (xy_hi, xy_hi, z_hi)`` from
``LORENZ_XY_BOX`` and ``LORENZ_Z_BOX``."""

LORENZ_FRAME_STREAM: int = 4003
"""Stream of the Lorenz frames, one ``3 x 3`` frame per initial condition, in order (D-081)."""

LYAP_PENDULUM_STREAM: int = 4004
"""Stream of the pendulum states of the agreement runs, drawn by the recipe of D-046(a): per
attempt ``rng.uniform(-CHAOTIC_THETA_MAX, CHAOTIC_THETA_MAX, size=2)`` with ``omega = 0``,
kept only if ``E > 0`` (D-082)."""

LYAP_PENDULUM_FRAME_STREAM: int = 4005
"""Stream of the pendulum frames, one ``4 x 4`` frame per state, in order (D-082)."""

STEP_PENDULUM_STREAM: int = 4006
"""Stream of the pendulum states of the per-step residual (D-079(b)): angles uniform in
``[-CHAOTIC_THETA_MAX, CHAOTIC_THETA_MAX]`` (one draw of shape ``(N_STEP_STATES, 2)``), then
velocities uniform in ``[-MOVING_OMEGA_MAX, MOVING_OMEGA_MAX]`` (a second draw). The same
states serve ``(theta, omega)`` and, through ``to_canonical``, ``(q, p)``."""

STEP_LORENZ_STREAM: int = 4007
"""Stream of the Lorenz points of the per-step residual (D-079(b)), drawn like the initial
conditions of ``LORENZ_IC_STREAM``."""

N_LORENZ_ICS: int = 4
"""Number of Lorenz initial conditions, each gated on its own (D-081)."""

N_LYAP_PENDULUM: int = 8
"""Number of pendulum states in the agreement runs (D-082)."""

N_STEP_STATES: int = 8
"""Number of states per system in the per-step residual gate (D-079(b), D-086)."""

LYAP_D0: float = 1e-9
"""Benettin's initial separation ``d0``, the brief's value (D-084)."""

LYAP_T: int = 1000
"""Averaging time of the pendulum, L4 and HO runs, in the time unit of the field (D-084). Their
step is ``AGREE_DT``."""

PENDULUM_T_TRANS: int = 100
"""Transient of the pendulum runs, declared a priori (D-083)."""

# The linear systems of D-080. ``A = V D V^-1`` with these matrices; the test writes the RK4-map
# references ``ln|R(dt mu)| / dt`` itself (D-080).

L4_D: tuple[tuple[float, ...], ...] = (
    (0.5, 0.0, 0.0, 0.0),
    (0.0, -0.2, 1.0, 0.0),
    (0.0, -1.0, -0.2, 0.0),
    (0.0, 0.0, 0.0, -1.0),
)
"""``D = blockdiag(0.5, [[-0.2, 1.0], [-1.0, -0.2]], -1.0)``, with the exponents
``(0.5, -0.2, -0.2, -1.0)``."""

L4_V: tuple[tuple[float, ...], ...] = (
    (1.0, 0.5, 0.0, 0.0),
    (0.0, 1.0, 0.5, 0.0),
    (0.0, 0.0, 1.0, 0.5),
    (0.5, 0.0, 0.0, 1.0),
)
"""Circulant ``V``, eigenvalues ``1.5``, ``1 +- 0.5i`` and ``0.5``, so ``kappa_2(V) = 3``."""

L4_CONTROL_EIGENVALUE: float = 0.6
"""The control of D-080 replaces the real eigenvalue ``0.5`` of ``D`` by this in the field."""

HO_OMEGA: float = 2.0
"""``Omega`` of the harmonic oscillator ``x' = v``, ``v' = -Omega**2 x`` (D-044, D-080)."""

HO_STATE0: tuple[float, float] = (1.0, 0.5)
"""Initial state of the HO runs (D-080)."""

L4_STATE0: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
"""Initial state of the L4 runs and their controls (D-080, review R3; in the register by D-089)."""

LINEAR_STEPS_PER_INTERVAL: int = 100
"""L4 and HO: ``tau_B = tau_Q = 1`` time unit, 100 steps of ``AGREE_DT`` (D-083)."""

# Lorenz (D-081).

LORENZ_SIGMA: float = 10.0
LORENZ_RHO: float = 28.0
LORENZ_BETA: float = 8.0 / 3.0
"""The classical parameters, those of the reference (D-081)."""

LORENZ_XY_BOX: tuple[float, float] = (-15.0, 15.0)
LORENZ_Z_BOX: tuple[float, float] = (10.0, 40.0)
"""Sampling box of the Lorenz initial conditions and per-step points (D-079(b), D-081)."""

LORENZ_DT: float = 1.0 / 128.0
"""Viswanath's step, so his finite-time bars apply to ours (D-081)."""

LORENZ_T: int = 10_000
"""Averaging time after the transient: the ``T = 10**4`` row of Viswanath's Table 5.2(c)
(D-081, Q4)."""

LORENZ_STEPS_PER_RENORM: int = 320
"""Lorenz ``tau_B = 2.5``: the largest divisor of the window ``1 280 000`` steps with
``lambda_1 tau_B <= ln 10`` at the published ``lambda_1`` (D-083)."""

LORENZ_LAMBDA1_REF: float = 0.90566
"""Viswanath (1998), PhD thesis, Cornell, Table 5.1: ``gamma_1`` (D-081). Literature, not run."""

LORENZ_LAMBDA1_REF_ERR: float = 0.00007
"""The reference's own maximum error, Table 5.1 (D-081)."""

LORENZ_LAMBDA1_BAR: float = 0.003
"""Viswanath's finite-time bar at ``T = 10**4``, Table 5.2(c). The gate is
``|lambda_1 - LORENZ_LAMBDA1_REF| <= LORENZ_LAMBDA1_BAR + LORENZ_LAMBDA1_REF_ERR`` (D-081)."""

LORENZ_LAMBDA2_TOL: float = 1e-3
"""``|lambda_2| <= LORENZ_LAMBDA2_TOL``: ten times Viswanath's ``10**-4`` reading at
``t = 10**4`` (D-081)."""

LORENZ_CONTROL_BETA_FACTOR: float = 1.01
"""The Lorenz control of D-079 multiplies ``beta`` by this in the field but not in ``ell``."""

# The exact-invariant gates (D-079, D-085, D-086).

KAPPA_CAP: float = 1e4
"""Validity cap of D-079(a): every recorded ``kappa_2(R_j)`` must be at most this, or the
identity fails as inconclusive."""

LOGDET_C: float = 100.0
"""Constant ``c`` of the rounding bound ``c * eps * steps_per_qr * n * kappa_j`` of D-079(a).
A declared allowance, not a proved constant, resting on the unproved assumption that the
intermediate frames are no worse conditioned than ``kappa_j`` (D-085, D-086; note section 2)."""

STEP_SLOPE_MIN: float = 4.0
"""The least-squares slope of ``log R(dt)`` against ``log dt`` must be at least this, where
``R(dt)`` is the largest ``|r|`` over the ``N_STEP_STATES`` states at each ``dt`` (D-079(b),
Q6, D-086 option 4). Per-state slopes and signs are reported, not gated."""

STEP_FLOOR_FACTOR: float = 1e3
"""Validity floor of D-079(b): ``R`` at the smallest ``dt`` must be at least
``STEP_FLOOR_FACTOR * eps``, about ``2.2e-13``, or the gate fails as inconclusive (D-086)."""

STEP_DT_GRID_THETA: tuple[float, float, float, float] = (0.08, 0.04, 0.02, 0.01)
"""Grid of the per-step residual for the pendulum in ``(theta, omega)`` (D-079(b))."""

STEP_DT_GRID_QP: tuple[float, float, float, float] = (0.16, 0.08, 0.04, 0.02)
"""Grid of the per-step residual for the pendulum in ``(q, p)``, declared a priori (D-085)."""

STEP_DT_GRID_LORENZ: tuple[float, float, float, float] = (0.005, 0.0025, 0.00125, 0.000625)
"""Grid of the per-step residual for Lorenz (D-079(b))."""

QP_CONTROL_DAMPING: float = 0.01
"""``gamma`` of the ``(q, p)`` control, ``p' <- p' - gamma p``, against the exact ``ell = 0``
(D-086)."""

# The agreement of Benettin and QR (D-082, D-085).

DELTA_BQ_S: float = 5.0
"""``s`` of the worst-case gate ``delta_BQ = s * eps * X / (d0 * dt)`` (D-085, Q7; note
section 5). It rests on two unproved assumptions, stated in the note."""

DELTA_BQ_Z: float = 5.0
"""Report only: ``z`` of the probabilistic form ``z * delta_BQ / sqrt(N_total)`` (D-085)."""

LOW_POWER_FRACTION: float = 0.1
"""Report only: a state is marked "low power" where ``delta_BQ > LOW_POWER_FRACTION *
|lambda_QR_1|`` (D-085)."""

X_CAP_ENERGY_ALLOWANCE: float = 0.01
"""Pendulum ``X`` cap: ``K_max = E(0) + E_s + X_CAP_ENERGY_ALLOWANCE * E_s``, a declared
allowance for RK4's energy drift (D-085; note section 6)."""

X_CAP_FACTOR: float = 1.01
"""Lorenz ``X`` cap: ``X_CAP_FACTOR * (sigma + rho + R)`` with the trapping-ball radius
``R = beta (sigma + rho) / (2 sqrt(beta - 1))``, a declared allowance for RK4 (D-085; note
section 6). Above a cap the agreement gate fails as inconclusive."""


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
