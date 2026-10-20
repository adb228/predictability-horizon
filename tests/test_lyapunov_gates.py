"""The milestone 5 gates and their negative controls (D-079 to D-082, D-085, D-086).

The references are exact invariants, closed forms or the literature, and every tolerance comes
from theory; all of them live in ``horizon.tolerances`` and were declared before any test ran. A
gate that fails is reported, never adjusted (D-025). The values the calibration lab sets
(D-083) are read from the register; until the lab's entry adds them, the tests that need them
fail with an ``AttributeError`` naming the value.

Each control runs a deliberately wrong input through the same measurement and must fail on the
gate's own rule, not on a validity rule: the controls match the message of the slope or of the
ceiling, so an ``inconclusive`` failure or a NaN cannot satisfy them. The one exception is the
three D-079(b) controls, which D-079 lets fail on the floor as weaker evidence (D-089). Every
measurement is made outside the ``raises`` block, so an exception from the code under test is an
error and not a pass. The measurements are in ``tests/lyapunov_checks.py``, shared with the script.
"""

import math
import re
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from horizon import lyapunov
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from tests import integrator_checks as chk
from tests import lyapunov_checks as lc

EXCEEDS = "exceeds the ceiling"
SLOPE_FAILS = r"slope: .* is below the floor"
FLOOR_FAILS = r"inconclusive, R = .* at the smallest dt is under the validity floor"
LORENZ_ICS = [pytest.param(i, id=f"ic-{i}") for i in range(tol.N_LORENZ_ICS)]
PENDULUM = [pytest.param(i, id=f"state-{i}") for i in range(tol.N_LYAP_PENDULUM)]


# ---- D-079(a): the bookkeeping identity, per QR interval -----------------------------------


@pytest.mark.parametrize("index", LORENZ_ICS)
def test_a_identity_holds_on_every_interval_of_the_lorenz_runs(index: int) -> None:
    """D-079(a): ``sum ln|R_ii| = log_det_steps`` within ``c eps N_Q n kappa_j``, every interval.

    Rounding only, along the run's own trajectory. Inconclusive if any ``kappa_j > KAPPA_CAP``.
    Catches a frame that is not the derivative of the step map the state follows, a QR whose
    ``|R_ii|`` are not the volume factors, and a lost interval. Blind to an error in ``rk4_step``
    itself, which (b) covers.
    """
    run = lc.lorenz_run(index)
    lc.check_logdet_identity(run.qr, lc.lab_value("LORENZ_STEPS_PER_QR"), what=f"Lorenz {index}")


@pytest.mark.parametrize("index", PENDULUM)
def test_a_identity_holds_on_every_interval_of_the_canonical_pendulum_runs(index: int) -> None:
    """D-079(a) on the ``(q, p)`` pendulum runs of D-082."""
    run = lc.pendulum_run(index)
    steps_per_qr = lc.lab_value("PENDULUM_STEPS_PER_QR")
    lc.check_logdet_identity(run.qr, steps_per_qr, what=f"pendulum (q, p) {index}")


@pytest.mark.parametrize("index", PENDULUM)
def test_a_identity_holds_on_every_interval_of_the_theta_omega_pendulum_runs(index: int) -> None:
    """D-079(a) on the ``(theta, omega)`` runs: with (b) in ``(theta, omega)``, the D-008 gate."""
    result = lc.pendulum_theta_qr(index)
    steps_per_qr = lc.lab_value("PENDULUM_STEPS_PER_QR")
    lc.check_logdet_identity(result, steps_per_qr, what=f"pendulum (theta, omega) {index}")


@pytest.mark.parametrize("system", ["l4", "ho"])
def test_a_identity_holds_on_every_interval_of_the_linear_runs(system: str) -> None:
    """D-079(a) on the L4 and HO runs of D-080, with the full frame."""
    run = lc.l4_run() if system == "l4" else lc.ho_run()
    lc.check_logdet_identity(run.qr, tol.LINEAR_STEPS_PER_INTERVAL, what=system)


@pytest.mark.parametrize("system", ["lorenz", "pendulum-qp"])
def test_a_control_an_euler_frame_fails_the_identity_bound(system: str) -> None:
    """Control of D-079(a): a frame propagated by the ``jvp`` of an Euler step, state and
    ``log_det_steps`` by RK4, through the same core. Must fail on the bound, with every
    ``kappa_j <= KAPPA_CAP``; an ``inconclusive`` failure does not satisfy it.
    """
    if system == "lorenz":
        field, params, dt = lc.lorenz, lc.LORENZ, tol.LORENZ_DT
        state0, frame = lc.lorenz_initial_conditions()[0], lc.lorenz_frames()[0]
        steps_per_qr = lc.lab_value("LORENZ_STEPS_PER_QR")
        transient, window = lc.lorenz_transient_steps(), lc.steps(tol.LORENZ_T, dt)
    else:
        field, params, dt = dp.canonical_vector_field, lc.DEFAULT, tol.AGREE_DT
        state0 = np.asarray(dp.to_canonical(jnp.asarray(lc.pendulum_states()[0]), lc.DEFAULT))
        frame = lc.pendulum_frames()[0]
        steps_per_qr = lc.lab_value("PENDULUM_STEPS_PER_QR")
        transient = lc.steps(tol.PENDULUM_T_TRANS, dt)
        window = lc.steps(tol.LYAP_T, dt)
    result = lc.qr_with(
        field,
        state0,
        params,
        frame,
        dt=dt,
        steps_per_qr=steps_per_qr,
        n_transient=lc.intervals(transient, steps_per_qr),
        n_qr=lc.intervals(window, steps_per_qr),
        tangent_step=lc.euler_tangent_step,
    )
    with pytest.raises(AssertionError, match=EXCEEDS):
        lc.check_logdet_identity(result, steps_per_qr, what=f"Euler-frame control, {system}")


# ---- D-079(b): the per-step residual, aggregate slope (D-086) ------------------------------


def _theta_step(ell: lc.Ell) -> lc.StepMeasurement:
    states = lc.step_pendulum_states()
    return lc.measure_step(dp.vector_field, lc.DEFAULT, states, tol.STEP_DT_GRID_THETA, ell)


def _canonical_step(field: lyapunov.Field) -> lc.StepMeasurement:
    states = lc.canonical_step_states()
    return lc.measure_step(field, lc.DEFAULT, states, tol.STEP_DT_GRID_QP, lc.ell_canonical)


def _lorenz_step(params: lc.LorenzParams) -> lc.StepMeasurement:
    points = lc.step_lorenz_points()
    return lc.measure_step(lc.lorenz, params, points, tol.STEP_DT_GRID_LORENZ, lc.ell_lorenz)


def test_b_theta_omega_residual_decays_at_order_four_or_more_the_d008_gate() -> None:
    """D-079(b) in ``(theta, omega)``, the D-008 gate with (a): ``ell = ln det M(theta_n) - ln det
    M(theta_{n+1})``. The slope of ``log R`` is at least 4.0, with ``R`` the largest ``|r|`` over
    the 8 states at each ``dt`` and the floor on ``R`` at the smallest ``dt`` (D-086). A wrong
    ``ell`` leaves an ``O(dt)`` residual; the controls below check that. Blind to a defect at
    only a few states, which the maximum can mask (D-086).
    """
    lc.check_step(_theta_step(lc.ell_theta), what="(theta, omega) per-step residual")


def test_b_canonical_residual_decays_at_order_four_or_more() -> None:
    """D-079(b) in ``(q, p)``: ``ell = 0``, on the ``(q, p)`` grid of D-085."""
    lc.check_step(_canonical_step(dp.canonical_vector_field), what="(q, p) per-step residual")


def test_b_lorenz_residual_decays_at_order_four_or_more() -> None:
    """D-079(b) for Lorenz: ``ell = -(sigma + 1 + beta) dt``."""
    lc.check_step(_lorenz_step(lc.LORENZ), what="Lorenz per-step residual")


@pytest.mark.parametrize(
    "control",
    [
        pytest.param("theta-ell-zero", id="theta-ell-zero"),
        pytest.param("theta-reciprocal", id="theta-reciprocal"),
        pytest.param("lorenz-beta", id="lorenz-beta"),
        pytest.param("qp-damping", id="qp-damping"),
    ],
)
def test_b_control_a_wrong_exact_value_fails_the_aggregate_slope(
    control: str, record_property: Callable[[str, object], None]
) -> None:
    """Controls of D-079(b) and D-086: each leaves an ``O(dt)`` residual, so the aggregate slope
    is about 1: D-008 with ``ell = 0`` and with the reciprocal formula; Lorenz with ``beta``
    changed by 1 percent in the field; ``(q, p)`` with a damping ``gamma = 0.01`` against
    ``ell = 0`` (D-086).

    The ``(q, p)`` control must fail on the slope rule (D-086). The three D-079 controls fail on
    the slope or on the floor, the floor being weaker evidence (D-079, D-089); a floor failure
    counts only with a finite ``R``, so a NaN does not satisfy them. The rule that triggered is
    recorded as the user property ``failed_rule``.
    """
    if control == "theta-ell-zero":
        measurement = _theta_step(lc.ell_canonical)
    elif control == "theta-reciprocal":
        measurement = _theta_step(lc.ell_theta_reciprocal)
    elif control == "lorenz-beta":
        measurement = _lorenz_step(lc.LORENZ_CONTROL)
    else:
        measurement = _canonical_step(lc.damped_canonical_field)
    accepted = SLOPE_FAILS if control == "qp-damping" else f"{SLOPE_FAILS}|{FLOOR_FAILS}"
    with pytest.raises(AssertionError, match=accepted) as failure:
        lc.check_step(measurement, what=f"control {control}")
    rule = "slope" if re.search(SLOPE_FAILS, str(failure.value)) else "floor"
    record_property("failed_rule", rule)
    record_property("failure_message", str(failure.value))
    if rule == "floor":
        assert math.isfinite(measurement.floor_value), f"control {control}: R is not finite"


# ---- D-080: the linear systems -------------------------------------------------------------


def _partial_sum(run: lc.Run, k: int) -> float:
    return float(np.sum(run.lambdas_qr[:k]))


@pytest.mark.parametrize("k", [1, 3, 4])
def test_l4_partial_sums_lie_within_the_finite_time_bound(k: int) -> None:
    """D-080, D-085: ``|S_k(T) - S_k| <= B_k`` for L4, with the RK4-map references.

    ``B_k = [k ln kappa(V) + ln(vol_k(C) / |det C_K|)] / T`` (note, section 1.2), computed here
    from the declared matrices and frame. A theorem, not a fitted number. Catches a missing or
    misplaced ``ln``, a wrong interval, a factor error and a frame that loses its orthogonality.
    """
    run = lc.l4_run()
    reference = float(
        np.sum(lc.rk4_map_exponents(lc.l4_continuous_eigenvalues(), tol.AGREE_DT)[:k])
    )
    bound = lc.linear_bound(lc.l4_matrix(), lc.l4_frame(), k, tol.LYAP_T)
    chk.check_at_most(abs(_partial_sum(run, k) - reference), bound, what=f"L4 S_{k}")


def test_ho_leading_exponent_lies_within_ln_kappa_over_t() -> None:
    """D-080, D-085: ``|lambda_1(T) - ref| <= ln kappa(V) / T = ln 2 / T`` for HO (tie rule)."""
    run = lc.ho_run()
    reference = float(lc.rk4_map_exponents(lc.ho_continuous_eigenvalues(), tol.AGREE_DT)[0])
    bound = lc.linear_bound(lc.ho_matrix(), lc.ho_frame(), 1, tol.LYAP_T)
    chk.check_at_most(abs(_partial_sum(run, 1) - reference), bound, what="HO S_1")


@pytest.mark.parametrize("system", ["l4", "ho"])
def test_linear_full_sums_match_the_rk4_map_on_every_interval(system: str) -> None:
    """D-080: the full sum per interval equals ``N_Q sum_i ln|R(dt mu_i)|``, exact per step,
    within the rounding bound of D-079(a), under its validity rule: inconclusive if any
    ``kappa_j > KAPPA_CAP`` (D-089)."""
    if system == "l4":
        run, mu = lc.l4_run(), lc.l4_continuous_eigenvalues()
    else:
        run, mu = lc.ho_run(), lc.ho_continuous_eigenvalues()
    lc.check_kappa_cap(run.qr, what=f"{system} full sum")
    n_q = tol.LINEAR_STEPS_PER_INTERVAL
    exact = n_q * float(np.sum(lc.rk4_map_exponents(mu, tol.AGREE_DT))) * tol.AGREE_DT
    log_r = np.asarray(run.qr.log_r)
    kappa = np.asarray(run.qr.cond_r)
    bound = tol.LOGDET_C * lc.EPS * n_q * log_r.shape[1] * kappa
    worst = float(np.max(np.abs(log_r.sum(axis=1) - exact) / bound))
    chk.check_at_most(worst, 1.0, what=f"{system} full sum against the RK4 map (bound = 1)")


@pytest.mark.parametrize("system", ["l4", "ho"])
def test_linear_benettin_lies_within_the_leading_bound(system: str) -> None:
    """D-080: Benettin's ``lambda`` within ``B_1``, ``direction0`` the frame's first column."""
    if system == "l4":
        run, a, frame, mu = (
            lc.l4_run(),
            lc.l4_matrix(),
            lc.l4_frame(),
            lc.l4_continuous_eigenvalues(),
        )
    else:
        run, a, frame, mu = (
            lc.ho_run(),
            lc.ho_matrix(),
            lc.ho_frame(),
            lc.ho_continuous_eigenvalues(),
        )
    reference = float(lc.rk4_map_exponents(mu, tol.AGREE_DT)[0])
    bound = lc.linear_bound(a, frame, 1, tol.LYAP_T)
    chk.check_at_most(abs(run.lambda_b - reference), bound, what=f"{system} Benettin")


def test_l4_control_a_shifted_eigenvalue_fails_the_bound() -> None:
    """Control of D-080: L4 with ``0.5`` replaced by ``0.6`` in the field, against the declared
    references. ``S_1`` is off by about 0.1, far above ``B_1``."""
    run = lc.linear_run(
        lc.l4_matrix(tol.L4_CONTROL_EIGENVALUE), np.array(tol.L4_STATE0), lc.l4_frame()
    )
    reference = float(lc.rk4_map_exponents(lc.l4_continuous_eigenvalues(), tol.AGREE_DT)[0])
    bound = lc.linear_bound(lc.l4_matrix(), lc.l4_frame(), 1, tol.LYAP_T)
    error = abs(_partial_sum(run, 1) - reference)
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(error, bound, what="control: L4 with a shifted eigenvalue")


def test_l4_control_qr_without_reorthonormalization_fails_s3() -> None:
    """Control of D-080: the frame is never replaced by ``Q``; ``S_3`` fails or is not finite."""
    n = lc.steps(tol.LYAP_T, tol.AGREE_DT)
    result = lc.qr_with(
        lc.linear,
        np.array(tol.L4_STATE0),
        jnp.asarray(lc.l4_matrix()),
        lc.l4_frame(),
        dt=tol.AGREE_DT,
        steps_per_qr=tol.LINEAR_STEPS_PER_INTERVAL,
        n_transient=0,
        n_qr=lc.intervals(n, tol.LINEAR_STEPS_PER_INTERVAL),
        orthonormalize=lc.keep_frame,
        step_logdet=False,
    )
    tau_q = tol.LINEAR_STEPS_PER_INTERVAL * tol.AGREE_DT
    s3 = float(np.sum(np.asarray(lyapunov.exponents(result.log_r, tau_q))[:3]))
    reference = float(
        np.sum(lc.rk4_map_exponents(lc.l4_continuous_eigenvalues(), tol.AGREE_DT)[:3])
    )
    bound = lc.linear_bound(lc.l4_matrix(), lc.l4_frame(), 3, tol.LYAP_T)
    with pytest.raises(AssertionError, match=f"{EXCEEDS}|not a finite number"):
        chk.check_at_most(abs(s3 - reference), bound, what="control: L4 without QR")


# ---- D-081: Lorenz against the literature --------------------------------------------------


@pytest.mark.parametrize("estimator", ["benettin", "qr"])
@pytest.mark.parametrize("index", LORENZ_ICS)
def test_lorenz_leading_exponent_lies_within_the_published_finite_time_bar(
    index: int, estimator: str
) -> None:
    """D-081: ``|lambda_1 - 0.90566| <= 0.003 + 0.00007`` at ``T = 10**4``, ``dt = 1/128``.

    Viswanath (1998), Tables 5.1 and 5.2(c). Each initial condition and estimator on its own; a
    failure is reported, never answered by a longer run (D-081).
    """
    run = lc.lorenz_run(index)
    value = run.lambda_b if estimator == "benettin" else float(run.lambdas_qr[0])
    chk.check_at_most(
        abs(value - tol.LORENZ_LAMBDA1_REF),
        tol.LORENZ_LAMBDA1_BAR + tol.LORENZ_LAMBDA1_REF_ERR,
        what=f"Lorenz {index} {estimator} lambda_1",
    )


@pytest.mark.parametrize("index", LORENZ_ICS)
def test_lorenz_second_exponent_is_zero_within_the_declared_tolerance(index: int) -> None:
    """D-081: ``|lambda_2| <= 1e-3``, ten times Viswanath's reading at ``t = 10**4``."""
    run = lc.lorenz_run(index)
    chk.check_at_most(
        abs(float(run.lambdas_qr[1])), tol.LORENZ_LAMBDA2_TOL, what=f"Lorenz {index} lambda_2"
    )


def test_lorenz_control_benettin_without_renormalization_falls_below_the_bar() -> None:
    """Control of D-081: one interval over the whole window; the separation saturates."""
    run = lc.lorenz_run(0)
    window = lc.steps(tol.LORENZ_T, tol.LORENZ_DT)
    result = lyapunov.benettin(
        lc.lorenz,
        run.qr.states[0],
        lc.LORENZ,
        jnp.asarray(lc.lorenz_frames()[0][:, 0]),
        dt=tol.LORENZ_DT,
        d0=tol.LYAP_D0,
        steps_per_renorm=window,
        n_transient=0,
        n_renorm=1,
        angle_mask=lc.NO_ANGLES_3,
    )
    value = float(lyapunov.exponents(result.log_growth, window * tol.LORENZ_DT))
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(
            abs(value - tol.LORENZ_LAMBDA1_REF),
            tol.LORENZ_LAMBDA1_BAR + tol.LORENZ_LAMBDA1_REF_ERR,
            what="control: Benettin without renormalization",
        )


def test_lorenz_control_qr_without_reorthonormalization_fails_lambda2() -> None:
    """Control of D-081: the frame is never replaced by ``Q``; ``lambda_2`` fails or is infinite."""
    steps_per_qr = lc.lab_value("LORENZ_STEPS_PER_QR")
    result = lc.qr_with(
        lc.lorenz,
        lc.lorenz_initial_conditions()[0],
        lc.LORENZ,
        lc.lorenz_frames()[0],
        dt=tol.LORENZ_DT,
        steps_per_qr=steps_per_qr,
        n_transient=lc.intervals(lc.lorenz_transient_steps(), steps_per_qr),
        n_qr=lc.intervals(lc.steps(tol.LORENZ_T, tol.LORENZ_DT), steps_per_qr),
        orthonormalize=lc.keep_frame,
        step_logdet=False,
    )
    value = float(np.asarray(lyapunov.exponents(result.log_r, steps_per_qr * tol.LORENZ_DT))[1])
    with pytest.raises(AssertionError, match=f"{EXCEEDS}|not a finite number"):
        chk.check_at_most(abs(value), tol.LORENZ_LAMBDA2_TOL, what="control: Lorenz without QR")


# ---- D-082: Benettin and QR agree by construction ------------------------------------------


@pytest.mark.parametrize("index", PENDULUM)
def test_benettin_and_qr_agree_on_the_canonical_pendulum_runs(index: int) -> None:
    """D-082, D-085: ``|lambda_B - lambda_QR,1| <= s eps X / (d0 dt)``, worst case, per state.

    Inconclusive above the energy cap on ``X``. On rotating states the bound is comparable to
    ``lambda_1`` and the gate has little power there (D-085); the script marks them.
    """
    run = lc.pendulum_run(index)
    lc.check_agreement(
        run.lambda_b,
        float(run.lambdas_qr[0]),
        float(run.benettin.max_abs_state),
        lc.pendulum_x_cap(lc.pendulum_states()[index]),
        tol.AGREE_DT,
        what=f"pendulum {index} agreement",
    )


@pytest.mark.parametrize("index", LORENZ_ICS)
def test_benettin_and_qr_agree_on_the_lorenz_runs(index: int) -> None:
    """D-082, D-085: the same gate on Lorenz, inconclusive above the trapping-ball cap."""
    run = lc.lorenz_run(index)
    lc.check_agreement(
        run.lambda_b,
        float(run.lambdas_qr[0]),
        float(run.benettin.max_abs_state),
        lc.lorenz_x_cap(),
        tol.LORENZ_DT,
        what=f"Lorenz {index} agreement",
    )


def test_agreement_control_qr_in_theta_omega_exceeds_delta_bq_on_some_state() -> None:
    """Control of D-082: QR on the ``(theta, omega)`` field against Benettin in ``(q, p)``, from
    the image of the same direction, must exceed ``delta_BQ`` on at least one state. Under the
    worst-case bound it may not (D-085); then it is reported as a finding, nothing tuned.
    """
    steps_per_qr = lc.lab_value("PENDULUM_STEPS_PER_QR")
    ratios = []
    for index in range(tol.N_LYAP_PENDULUM):
        run = lc.pendulum_run(index)
        theta = lc.pendulum_theta_qr(index)
        lambda_theta = float(
            np.asarray(lyapunov.exponents(theta.log_r, steps_per_qr * tol.AGREE_DT))[0]
        )
        x = float(run.benettin.max_abs_state)
        x_cap = lc.pendulum_x_cap(lc.pendulum_states()[index])
        lc.check_x_cap(x, x_cap, what=f"control: coordinate mismatch, state {index}")  # D-089
        ratios.append(abs(run.lambda_b - lambda_theta) / lc.delta_bq(x, tol.AGREE_DT))
    with pytest.raises(AssertionError, match=EXCEEDS):
        chk.check_at_most(max(ratios), 1.0, what="control: coordinate mismatch (bound = 1)")
