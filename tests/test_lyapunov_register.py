"""Tripwire for the milestone 5 register (K1) and its seeded-case guards (C).

K1 pins the declared values in ``src/horizon/tolerances.py`` (D-079 to D-086). The C tests check
that the seeded cases of ``tests/lyapunov_checks.py`` follow the recipes of the log, by
recomputing each recipe here from its stream, independently of that module, and check the
derived formulas (the caps on ``X``, the bound ``B_k`` and the RK4-map references) against
values written out by hand from the note. None of these tests integrates anything.
"""

import math

import numpy as np

from horizon import tolerances as tol
from tests import lyapunov_checks as lc

# ---- K1: the register ---------------------------------------------------------------------


def test_register_holds_the_declared_lyapunov_values() -> None:
    """K1 (class exact). Tripwire: these literals are the values logged in D-079 to D-086.

    Catches: an edit to the milestone 5 section of ``src/horizon/tolerances.py`` without a
    matching edit here, so every change shows up in two places in the diff. Changing a value
    needs the owner's approval first (D-025). Blind to: a declared value that is itself wrong,
    which is what the design review is for.
    """
    # D-084: the streams and the counts.
    assert (tol.L4_FRAME_STREAM, tol.HO_FRAME_STREAM) == (4000, 4001)
    assert (tol.LORENZ_IC_STREAM, tol.LORENZ_FRAME_STREAM) == (4002, 4003)
    assert (tol.LYAP_PENDULUM_STREAM, tol.LYAP_PENDULUM_FRAME_STREAM) == (4004, 4005)
    assert (tol.STEP_PENDULUM_STREAM, tol.STEP_LORENZ_STREAM) == (4006, 4007)
    assert (tol.N_LORENZ_ICS, tol.N_LYAP_PENDULUM, tol.N_STEP_STATES) == (4, 8, 8)

    # D-084: the values declared a priori.
    assert (tol.LYAP_D0, tol.AGREE_DT, tol.LYAP_T, tol.PENDULUM_T_TRANS) == (1e-9, 0.01, 1000, 100)
    assert tol.LINEAR_STEPS_PER_INTERVAL == 100

    # D-080: the linear systems and their control.
    assert tol.L4_D == (
        (0.5, 0.0, 0.0, 0.0),
        (0.0, -0.2, 1.0, 0.0),
        (0.0, -1.0, -0.2, 0.0),
        (0.0, 0.0, 0.0, -1.0),
    )
    assert tol.L4_V == (
        (1.0, 0.5, 0.0, 0.0),
        (0.0, 1.0, 0.5, 0.0),
        (0.0, 0.0, 1.0, 0.5),
        (0.5, 0.0, 0.0, 1.0),
    )
    assert (tol.L4_CONTROL_EIGENVALUE, tol.HO_OMEGA, tol.HO_STATE0) == (0.6, 2.0, (1.0, 0.5))
    assert tol.L4_STATE0 == (0.0, 0.0, 0.0, 0.0)  # D-089

    # D-081, D-083: Lorenz.
    assert (tol.LORENZ_SIGMA, tol.LORENZ_RHO, tol.LORENZ_BETA) == (10.0, 28.0, 8.0 / 3.0)
    assert (tol.LORENZ_XY_BOX, tol.LORENZ_Z_BOX) == ((-15.0, 15.0), (10.0, 40.0))
    assert (tol.LORENZ_DT, tol.LORENZ_T, tol.LORENZ_STEPS_PER_RENORM) == (1 / 128, 10_000, 320)
    assert (tol.LORENZ_LAMBDA1_REF, tol.LORENZ_LAMBDA1_REF_ERR) == (0.90566, 0.00007)
    assert (tol.LORENZ_LAMBDA1_BAR, tol.LORENZ_LAMBDA2_TOL) == (0.003, 1e-3)
    assert tol.LORENZ_CONTROL_BETA_FACTOR == 1.01

    # D-079, D-085, D-086: the exact-invariant gates.
    assert (tol.KAPPA_CAP, tol.LOGDET_C) == (1e4, 100.0)
    assert (tol.STEP_SLOPE_MIN, tol.STEP_FLOOR_FACTOR) == (4.0, 1e3)
    assert tol.STEP_DT_GRID_THETA == (0.08, 0.04, 0.02, 0.01)
    assert tol.STEP_DT_GRID_QP == (0.16, 0.08, 0.04, 0.02)
    assert tol.STEP_DT_GRID_LORENZ == (0.005, 0.0025, 0.00125, 0.000625)
    assert tol.QP_CONTROL_DAMPING == 0.01

    # D-085: the agreement gate.
    assert (tol.DELTA_BQ_S, tol.DELTA_BQ_Z, tol.LOW_POWER_FRACTION) == (5.0, 5.0, 0.1)
    assert (tol.X_CAP_ENERGY_ALLOWANCE, tol.X_CAP_FACTOR) == (0.01, 1.01)

    # D-091, D-092: the calibration lab's values.
    assert (tol.PENDULUM_STEPS_PER_QR, tol.PENDULUM_STEPS_PER_RENORM) == (80, 400)
    assert (tol.LORENZ_STEPS_PER_QR, tol.LORENZ_TRANSIENT_STEPS) == (25, 1600)


def test_register_streams_are_distinct_and_disjoint_from_the_earlier_ones() -> None:
    """K1 (class exact): streams 4000 to 4007 are distinct and reuse no earlier stream (D-084)."""
    streams = (
        tol.L4_FRAME_STREAM,
        tol.HO_FRAME_STREAM,
        tol.LORENZ_IC_STREAM,
        tol.LORENZ_FRAME_STREAM,
        tol.LYAP_PENDULUM_STREAM,
        tol.LYAP_PENDULUM_FRAME_STREAM,
        tol.STEP_PENDULUM_STREAM,
        tol.STEP_LORENZ_STREAM,
    )
    assert sorted(streams) == list(range(4000, 4008))
    earlier = {*range(20), 1000, *range(2000, 2004), *range(3000, 3007)}
    assert earlier.isdisjoint(streams)


def test_register_windows_divide_into_whole_intervals() -> None:
    """K1 (class exact): every a-priori window is a whole number of steps and intervals (D-083).

    The Lorenz window is ``1 280 000 = 2**11 * 5**4`` steps, which 320 divides; the pendulum,
    L4 and HO windows are ``100 000`` steps, which ``LINEAR_STEPS_PER_INTERVAL`` divides. The
    Lorenz ``tau_B`` keeps ``lambda_1 tau_B <= ln 10`` at the published ``lambda_1``.
    """
    lorenz_window = lc.steps(tol.LORENZ_T, tol.LORENZ_DT)
    assert lorenz_window == 1_280_000
    assert lorenz_window % tol.LORENZ_STEPS_PER_RENORM == 0
    tau_b = tol.LORENZ_STEPS_PER_RENORM * tol.LORENZ_DT
    assert tol.LORENZ_LAMBDA1_REF * tau_b <= math.log(10.0)
    assert lc.steps(tol.LYAP_T, tol.AGREE_DT) % tol.LINEAR_STEPS_PER_INTERVAL == 0
    assert lc.steps(tol.PENDULUM_T_TRANS, tol.AGREE_DT) == 10_000


# ---- C: the seeded cases follow the logged recipes ------------------------------------------


def _frame(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q @ np.diag(np.sign(np.diag(r)))


def test_frames_follow_the_recipe_and_are_orthonormal() -> None:
    """C (class exact): each frame is the signed ``Q`` of a Gaussian matrix from its stream."""
    for stream, n, count, got in (
        (tol.L4_FRAME_STREAM, 4, 1, lc.l4_frame()[None]),
        (tol.HO_FRAME_STREAM, 2, 1, lc.ho_frame()[None]),
        (tol.LORENZ_FRAME_STREAM, 3, tol.N_LORENZ_ICS, lc.lorenz_frames()),
        (tol.LYAP_PENDULUM_FRAME_STREAM, 4, tol.N_LYAP_PENDULUM, lc.pendulum_frames()),
    ):
        rng = np.random.default_rng([tol.CASES_SEED, stream])
        want = np.stack([_frame(rng, n) for _ in range(count)])
        np.testing.assert_array_equal(got, want)
        for frame in got:
            assert np.max(np.abs(frame.T @ frame - np.eye(n))) < 1e-14


def test_lorenz_points_follow_the_recipe() -> None:
    """C (class exact): one uniform draw of shape ``(N, 3)`` in the declared box, per stream."""
    for stream, count, got in (
        (tol.LORENZ_IC_STREAM, tol.N_LORENZ_ICS, lc.lorenz_initial_conditions()),
        (tol.STEP_LORENZ_STREAM, tol.N_STEP_STATES, lc.step_lorenz_points()),
    ):
        rng = np.random.default_rng([tol.CASES_SEED, stream])
        want = rng.uniform([-15.0, -15.0, 10.0], [15.0, 15.0, 40.0], size=(count, 3))
        np.testing.assert_array_equal(got, want)


def test_pendulum_states_follow_their_recipes() -> None:
    """C (class exact): rest states with ``E > 0`` (D-046(a)); moving per-step states (D-079)."""
    rng = np.random.default_rng([tol.CASES_SEED, tol.LYAP_PENDULUM_STREAM])
    kept = []
    while len(kept) < tol.N_LYAP_PENDULUM:
        theta = rng.uniform(-math.pi, math.pi, size=2)
        if -2.0 * math.cos(theta[0]) - math.cos(theta[1]) > 0.0:
            kept.append([theta[0], theta[1], 0.0, 0.0])
    np.testing.assert_array_equal(lc.pendulum_states(), np.array(kept))

    rng = np.random.default_rng([tol.CASES_SEED, tol.STEP_PENDULUM_STREAM])
    theta = rng.uniform(-math.pi, math.pi, size=(tol.N_STEP_STATES, 2))
    omega = rng.uniform(-2.0, 2.0, size=(tol.N_STEP_STATES, 2))
    np.testing.assert_array_equal(lc.step_pendulum_states(), np.hstack([theta, omega]))


def test_the_derived_formulas_match_the_note() -> None:
    """C (class exact up to rounding): the caps, ``kappa(V)`` and the HO bound, by hand.

    Lorenz: ``R = 38 (8/3) / (2 sqrt(5/3)) ~ 39.25`` and ``X_cap = 1.01 (38 + R) ~ 78.0``. L4:
    ``kappa_2(V) = 3`` (circulant, eigenvalue moduli 1.5 and 0.5). HO: ``kappa(V) = 2``, so
    ``B_1 = ln 2 / 1000`` by the tie rule (note, section 1.1). Pendulum, default parameters
    (``E_s = 3``, ``tr M = 3``, ``m1 m2 L1**2 L2**2 = 1``): ``K_max = E(0) + 3.03`` and both
    ``p_max`` and ``omega_max`` are ``sqrt(6 K_max)``.
    """
    radius = 38.0 * (8.0 / 3.0) / (2.0 * math.sqrt(5.0 / 3.0))
    assert abs(lc.lorenz_x_cap() - 1.01 * (38.0 + radius)) < 1e-12
    assert 77.9 < lc.lorenz_x_cap() < 78.1
    _, v = lc.unit_eigenvectors(lc.l4_matrix())
    assert abs(np.linalg.cond(v, 2) - 3.0) < 1e-12
    assert (
        abs(lc.linear_bound(lc.ho_matrix(), lc.ho_frame(), 1, 1000.0) - math.log(2.0) / 1000)
        < 1e-15
    )
    state = np.array([math.pi / 2, 0.0, 0.0, 0.0])
    e0 = -2.0 * math.cos(state[0]) - math.cos(state[1])
    k_max = e0 + 3.0 + 0.03
    want = max(math.pi / 2 + 1100.0 * math.sqrt(2.0 * k_max * 3.0), math.sqrt(2.0 * k_max * 3.0))
    assert abs(lc.pendulum_x_cap(state) - want) < 1e-9


def test_rk4_map_references_are_the_closed_forms() -> None:
    """C: HO's ``ln|R(dt mu)| / dt`` is below 0, from ``|R(iy)|**2 = 1 - y**6/72 + y**8/576``."""
    y = tol.HO_OMEGA * tol.AGREE_DT
    want = 0.5 * math.log1p(-(y**6) / 72.0 + y**8 / 576.0) / tol.AGREE_DT
    got = lc.rk4_map_exponents(lc.ho_continuous_eigenvalues(), tol.AGREE_DT)
    # 2 Re w + |w|**2 cancels from about y**2 to y**6 / 72, so the helper is good to about
    # eps * y**2 / (y**6 / 72), about 5e-8 relative; 1e-6 is far below the gate's ln 2 / T.
    assert np.all(np.abs(got - want) <= 1e-6 * abs(want)) and want < 0.0
