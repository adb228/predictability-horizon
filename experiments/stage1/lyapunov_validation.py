"""Print the milestone 5 Lyapunov validation: every gate statistic, its bound and its controls.

Run from the repository root::

    uv run python -m experiments.stage1.lyapunov_validation

It writes no files and makes no figures (D-076). It uses the same runs, measurements and check
functions as the tests (``tests/lyapunov_checks.py``, ``tests/integrator_checks.py``) and the
declared values of ``horizon.tolerances``, so each number printed here is the number its gate
judged, printed next to its bound. Two short computations of ``tests/test_lyapunov_gates.py``
(the partial sums and the full-sum ratio of D-080) are repeated here, line for line, because
the test module is not importable as a library. Nothing is chosen here. Each negative control
must fail on its gate's own rule; a control that passes, or fails on another rule, counts as a
failure. The exit status is 1 if any check fails and 0 otherwise.

The items marked "report only" are the ones D-079, D-082, D-085 and D-086 ask the script to
print without a gate. The running sums are printed at the powers of ten up to ``T`` that fall
on a QR boundary of the run (``t = 100, 1000`` for the pendulum, ``t = 100, 1000, 10**4`` for
Lorenz). Those times are a display choice, not a register value, and judge nothing.

A number printed by this script is a measurement made at the commit named in the provenance
header, which must say "clean" for the printout to be pasted into the decision log (D-060).
"""

import math
import re
import sys
import time
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from experiments.stage1 import integrator_validation as m3
from horizon import lyapunov
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from tests import integrator_checks as chk
from tests import lyapunov_checks as lc

EXCEEDS = "exceeds the ceiling"
SLOPE_FAILS = r"slope: .* is below the floor"
FLOOR_FAILS = r"inconclusive, R = .* at the smallest dt is under the validity floor"
NOT_FINITE = "not a finite number"


def _seconds(start: float) -> str:
    return f"({time.perf_counter() - start:.0f} s)"


def _control(
    failures: list[str],
    label: str,
    rules: dict[str, str],
    check: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> str:
    """Run a control's check, which must raise on one of ``rules`` (name to message pattern).

    Return the verdict: the rule it failed on, or a failure recorded under ``label`` if it
    passed or failed on another rule.
    """
    try:
        check(*args, **kwargs)
    except AssertionError as error:
        for name, pattern in rules.items():
            if re.search(pattern, str(error)):
                return f"fails as required ({name})"
        failures.append(f"{label}: the control failed on another rule: {error}")
        return "FAIL (another rule)"
    failures.append(f"{label}: the control did not fail")
    return "FAIL (did not fail)"


def _checkpoints(horizon: float, interval: float) -> list[tuple[float, int]]:
    """Return the powers of ten up to ``horizon`` on a boundary of ``interval``, with counts.

    Each entry is ``(t, intervals to t)``.
    """
    points = []
    for k in range(int(math.log10(horizon)) + 1):
        t = 10.0**k
        count = round(t / interval)
        if count > 0 and abs(count * interval - t) <= 1e-9 * t:
            points.append((t, count))
    return points


def _ratio(value: float, bound: float) -> str:
    return f"{value / bound:.3g}" if math.isfinite(value) else "n/a"


def _partial_sum(run: lc.Run, k: int) -> float:
    return float(np.sum(run.lambdas_qr[:k]))


# ---- 1. D-079(a) ---------------------------------------------------------------------------


def _euler_control(system: str) -> tuple[lyapunov.QRResult, int]:
    """Return the Euler-frame control run, as in ``tests/test_lyapunov_gates.py``."""
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
    return result, steps_per_qr


def report_identity(failures: list[str]) -> None:
    """Section 1: the bookkeeping identity of D-079(a), per QR interval."""
    start = time.perf_counter()
    print(
        "\n1. The identity of D-079(a): per QR interval, |sum ln|R_ii| - log_det_steps| over "
        "c eps N_Q n kappa_j,\n"
        f"   c = {tol.LOGDET_C:g}; the gate is the worst ratio <= 1, valid if every kappa_j <= "
        f"KAPPA_CAP = {tol.KAPPA_CAP:.0e}"
    )
    print(
        f"   {'run':<26}{'N_Q':>5}{'intervals':>10}{'max kappa_j':>13}{'worst ratio':>13}  verdict"
    )
    lorenz_q = lc.lab_value("LORENZ_STEPS_PER_QR")
    pendulum_q = lc.lab_value("PENDULUM_STEPS_PER_QR")
    rows: list[tuple[str, lyapunov.QRResult, int]] = []
    rows += [(f"Lorenz {i}", lc.lorenz_run(i).qr, lorenz_q) for i in range(tol.N_LORENZ_ICS)]
    rows += [
        (f"pendulum (q, p) {i}", lc.pendulum_run(i).qr, pendulum_q)
        for i in range(tol.N_LYAP_PENDULUM)
    ]
    rows += [
        (f"pendulum (theta, omega) {i}", lc.pendulum_theta_qr(i), pendulum_q)
        for i in range(tol.N_LYAP_PENDULUM)
    ]
    rows += [
        ("L4", lc.l4_run().qr, tol.LINEAR_STEPS_PER_INTERVAL),
        ("HO", lc.ho_run().qr, tol.LINEAR_STEPS_PER_INTERVAL),
    ]
    for name, result, steps_per_qr in rows:
        ratios = lc.logdet_identity_ratios(result, steps_per_qr)
        worst = float(np.max(ratios))
        kappa = float(np.max(np.asarray(result.cond_r)))
        verdict = m3._judge(
            failures,
            f"D-079(a) {name}",
            lc.check_logdet_identity,
            result,
            steps_per_qr,
            what=name,
        )
        print(
            f"   {name:<26}{steps_per_qr:>5}{len(ratios):>10}{kappa:>13.1f}{worst:>13.3e}  "
            f"{verdict}"
        )
    print("   Control: the frame follows the jvp of an Euler step; must fail on the bound")
    for system in ("lorenz", "pendulum-qp"):
        result, steps_per_qr = _euler_control(system)
        ratios = lc.logdet_identity_ratios(result, steps_per_qr)
        kappa = float(np.max(np.asarray(result.cond_r)))
        verdict = _control(
            failures,
            f"D-079(a) Euler-frame control, {system}",
            {"bound": EXCEEDS},
            lc.check_logdet_identity,
            result,
            steps_per_qr,
            what=f"Euler-frame control, {system}",
        )
        print(
            f"   {'Euler frame, ' + system:<26}{steps_per_qr:>5}{len(ratios):>10}{kappa:>13.1f}"
            f"{float(np.max(ratios)):>13.3e}  {verdict}"
        )
    print(f"   section 1 took {_seconds(start)}")


# ---- 2. D-079(b), D-086 --------------------------------------------------------------------


def _step_cases() -> list[tuple[str, lc.StepMeasurement]]:
    theta = lc.step_pendulum_states()
    canonical = lc.canonical_step_states()
    lorenz = lc.step_lorenz_points()
    return [
        (
            "(theta, omega)",
            lc.measure_step(
                dp.vector_field, lc.DEFAULT, theta, tol.STEP_DT_GRID_THETA, lc.ell_theta
            ),
        ),
        (
            "(q, p)",
            lc.measure_step(
                dp.canonical_vector_field,
                lc.DEFAULT,
                canonical,
                tol.STEP_DT_GRID_QP,
                lc.ell_canonical,
            ),
        ),
        (
            "Lorenz",
            lc.measure_step(lc.lorenz, lc.LORENZ, lorenz, tol.STEP_DT_GRID_LORENZ, lc.ell_lorenz),
        ),
    ]


def _step_controls() -> list[tuple[str, lc.StepMeasurement, dict[str, str]]]:
    theta = lc.step_pendulum_states()
    canonical = lc.canonical_step_states()
    slope_or_floor = {"slope": SLOPE_FAILS, "floor": FLOOR_FAILS}
    return [
        (
            "theta-ell-zero",
            lc.measure_step(
                dp.vector_field, lc.DEFAULT, theta, tol.STEP_DT_GRID_THETA, lc.ell_canonical
            ),
            slope_or_floor,
        ),
        (
            "theta-reciprocal",
            lc.measure_step(
                dp.vector_field, lc.DEFAULT, theta, tol.STEP_DT_GRID_THETA, lc.ell_theta_reciprocal
            ),
            slope_or_floor,
        ),
        (
            "lorenz-beta",
            lc.measure_step(
                lc.lorenz,
                lc.LORENZ_CONTROL,
                lc.step_lorenz_points(),
                tol.STEP_DT_GRID_LORENZ,
                lc.ell_lorenz,
            ),
            slope_or_floor,
        ),
        (
            "qp-damping",
            lc.measure_step(
                lc.damped_canonical_field,
                lc.DEFAULT,
                canonical,
                tol.STEP_DT_GRID_QP,
                lc.ell_canonical,
            ),
            {"slope": SLOPE_FAILS},
        ),
    ]


def report_step(failures: list[str]) -> None:
    """Section 2: the per-step residual of D-079(b), the aggregate slope of D-086."""
    start = time.perf_counter()
    floor = tol.STEP_FLOOR_FACTOR * lc.EPS
    print(
        "\n2. The per-step residual of D-079(b) and D-086: R(dt) = max over the 8 states of "
        "|ln|det J_step| - ell|;\n"
        f"   the gate is the slope of log R >= {tol.STEP_SLOPE_MIN}, valid if R at the smallest "
        f"dt >= {floor:.3e}"
    )
    for name, m in _step_cases():
        verdict = m3._judge(failures, f"D-079(b) {name}", lc.check_step, m, what=name)
        grid = ", ".join(f"{h:g}" for h in m.grid)
        values = ", ".join(f"{r:.3e}" for r in m.largest)
        print(f"   {name}: dt = {grid}")
        print(f"      R = {values}")
        print(
            f"      slope {m.slope:.4f} (floor {tol.STEP_SLOPE_MIN}), R at the smallest dt "
            f"{m.floor_value:.3e} = {m.floor_value / floor:.3g} times the validity floor  {verdict}"
        )
        slopes = m.state_slopes
        changed = [i for i, one in enumerate(m.state_one_sign) if not one]
        print(
            f"      report only: per-state slopes {slopes.min():.3f} to {slopes.max():.3f}; "
            + (
                f"r changes sign over the grid at state(s) {changed}"
                if changed
                else "r keeps one sign over the grid at every state"
            )
        )
    print("   Controls: an O(dt) residual; must fail on the slope (or the floor, D-079, D-089)")
    for name, m, rules in _step_controls():
        verdict = _control(failures, f"D-079(b) control {name}", rules, lc.check_step, m, what=name)
        if verdict.startswith("fails") and "floor" in verdict and not math.isfinite(m.floor_value):
            failures.append(f"D-079(b) control {name}: R is not finite")
            verdict = "FAIL (R not finite)"
        print(
            f"   {name:<17} slope {m.slope:.4f}, R at the smallest dt {m.floor_value:.3e}  "
            f"{verdict}"
        )
    print(f"   section 2 took {_seconds(start)}")


# ---- 3. D-080 ------------------------------------------------------------------------------


def report_linear(failures: list[str]) -> None:
    """Section 3: L4 and HO against the RK4-map references and the finite-time bounds."""
    start = time.perf_counter()
    print(
        f"\n3. The linear systems of D-080: T = {tol.LYAP_T}, dt = {tol.AGREE_DT}, "
        f"tau_B = tau_Q = {tol.LINEAR_STEPS_PER_INTERVAL * tol.AGREE_DT:g}; references are the "
        "exponents of the RK4 map"
    )
    l4, ho = lc.l4_run(), lc.ho_run()
    l4_ref = lc.rk4_map_exponents(lc.l4_continuous_eigenvalues(), tol.AGREE_DT)
    ho_ref = lc.rk4_map_exponents(lc.ho_continuous_eigenvalues(), tol.AGREE_DT)
    print(f"   {'statistic':<34}{'value':>12}{'bound':>12}{'ratio':>9}  verdict")
    rows: list[tuple[str, float, float]] = []
    for k in (1, 3, 4):
        error = abs(_partial_sum(l4, k) - float(np.sum(l4_ref[:k])))
        rows.append(
            (
                f"L4 |S_{k} - ref| <= B_{k}",
                error,
                lc.linear_bound(lc.l4_matrix(), lc.l4_frame(), k, tol.LYAP_T),
            )
        )
    rows.append(
        (
            "HO |S_1 - ref| <= ln kappa(V) / T",
            abs(_partial_sum(ho, 1) - float(ho_ref[0])),
            lc.linear_bound(lc.ho_matrix(), lc.ho_frame(), 1, tol.LYAP_T),
        )
    )
    rows.append(
        (
            "L4 Benettin |lambda - ref| <= B_1",
            abs(l4.lambda_b - float(l4_ref[0])),
            lc.linear_bound(lc.l4_matrix(), lc.l4_frame(), 1, tol.LYAP_T),
        )
    )
    rows.append(
        (
            "HO Benettin |lambda - ref| <= B_1",
            abs(ho.lambda_b - float(ho_ref[0])),
            lc.linear_bound(lc.ho_matrix(), lc.ho_frame(), 1, tol.LYAP_T),
        )
    )
    for name, value, bound in rows:
        verdict = m3._judge(failures, f"D-080 {name}", chk.check_at_most, value, bound, what=name)
        print(f"   {name:<34}{value:>12.3e}{bound:>12.3e}{_ratio(value, bound):>9}  {verdict}")
    n_q = tol.LINEAR_STEPS_PER_INTERVAL
    for system, run, mu in (
        ("L4", l4, lc.l4_continuous_eigenvalues()),
        ("HO", ho, lc.ho_continuous_eigenvalues()),
    ):
        cap = m3._judge(
            failures, f"D-080 {system} full sum, kappa cap", lc.check_kappa_cap, run.qr, what=system
        )
        exact = n_q * float(np.sum(lc.rk4_map_exponents(mu, tol.AGREE_DT))) * tol.AGREE_DT
        log_r = np.asarray(run.qr.log_r)
        kappa = np.asarray(run.qr.cond_r)
        bound = tol.LOGDET_C * lc.EPS * n_q * log_r.shape[1] * kappa
        worst = float(np.max(np.abs(log_r.sum(axis=1) - exact) / bound))
        verdict = m3._judge(
            failures,
            f"D-080 {system} full sum",
            chk.check_at_most,
            worst,
            1.0,
            what=f"{system} full sum against the RK4 map (bound = 1)",
        )
        name = f"{system} full sum per interval (ratio)"
        print(
            f"   {name:<34}{worst:>12.3e}{1.0:>12.3e}{_ratio(worst, 1.0):>9}  {verdict}; "
            f"max kappa_j {float(np.max(kappa)):.2f} {cap}"
        )
    print("   report only: the QR exponents against the RK4-map references")
    for system, run, ref in (("L4", l4, l4_ref), ("HO", ho, ho_ref)):
        cells = "  ".join(f"{v:+.6f} ({r:+.6f})" for v, r in zip(run.lambdas_qr, ref, strict=True))
        print(f"   {system}: {cells}")
    print("   Controls: must fail on the bound (or be not finite, for the run without QR)")
    run = lc.linear_run(
        lc.l4_matrix(tol.L4_CONTROL_EIGENVALUE), np.array(tol.L4_STATE0), lc.l4_frame()
    )
    bound = lc.linear_bound(lc.l4_matrix(), lc.l4_frame(), 1, tol.LYAP_T)
    error = abs(_partial_sum(run, 1) - float(l4_ref[0]))
    verdict = _control(
        failures,
        "D-080 control, shifted eigenvalue",
        {"bound": EXCEEDS},
        chk.check_at_most,
        error,
        bound,
        what="control: L4 with a shifted eigenvalue",
    )
    name = f"L4 eigenvalue {tol.L4_CONTROL_EIGENVALUE}, |S_1 - ref|"
    print(f"   {name:<34}{error:>12.3e}{bound:>12.3e}{_ratio(error, bound):>9}  {verdict}")
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
    bound = lc.linear_bound(lc.l4_matrix(), lc.l4_frame(), 3, tol.LYAP_T)
    error = abs(s3 - float(np.sum(l4_ref[:3])))
    verdict = _control(
        failures,
        "D-080 control, no re-orthonormalization",
        {"bound": EXCEEDS, "not finite": NOT_FINITE},
        chk.check_at_most,
        error,
        bound,
        what="control: L4 without QR",
    )
    name = "L4 without QR, |S_3 - ref|"
    print(f"   {name:<34}{error:>12.3e}{bound:>12.3e}{_ratio(error, bound):>9}  {verdict}")
    print(f"   section 3 took {_seconds(start)}")


# ---- 4. D-081 ------------------------------------------------------------------------------


def report_lorenz(failures: list[str]) -> None:
    """Section 4: Lorenz against the literature, both estimators per initial condition."""
    start = time.perf_counter()
    bar = tol.LORENZ_LAMBDA1_BAR + tol.LORENZ_LAMBDA1_REF_ERR
    print(
        f"\n4. Lorenz against Viswanath (D-081): T = {tol.LORENZ_T}, "
        f"dt = 1/{round(1 / tol.LORENZ_DT)}, "
        f"transient {lc.lorenz_transient_steps()} steps, tau_B = {tol.LORENZ_STEPS_PER_RENORM} "
        f"steps, tau_Q = {lc.lab_value('LORENZ_STEPS_PER_QR')} steps"
    )
    print(
        f"   |lambda_1 - {tol.LORENZ_LAMBDA1_REF}| <= {tol.LORENZ_LAMBDA1_BAR} + "
        f"{tol.LORENZ_LAMBDA1_REF_ERR} = {bar:.5f}; |lambda_2| <= {tol.LORENZ_LAMBDA2_TOL:g}"
    )
    print(
        f"   {'ic':<4}{'statistic':<26}{'value':>12}{'|error|':>12}{'bound':>10}{'ratio':>9}"
        "  verdict"
    )
    for i in range(tol.N_LORENZ_ICS):
        run = lc.lorenz_run(i)
        lambdas = run.lambdas_qr
        for name, value in (
            ("lambda_1 Benettin", run.lambda_b),
            ("lambda_1 QR", float(lambdas[0])),
        ):
            error = abs(value - tol.LORENZ_LAMBDA1_REF)
            verdict = m3._judge(
                failures,
                f"D-081 Lorenz {i} {name}",
                chk.check_at_most,
                error,
                bar,
                what=f"Lorenz {i} {name}",
            )
            print(
                f"   {i:<4}{name:<26}{value:>12.6f}{error:>12.3e}{bar:>10.5f}"
                f"{_ratio(error, bar):>9}  {verdict}"
            )
        value = float(lambdas[1])
        verdict = m3._judge(
            failures,
            f"D-081 Lorenz {i} lambda_2",
            chk.check_at_most,
            abs(value),
            tol.LORENZ_LAMBDA2_TOL,
            what=f"Lorenz {i} lambda_2",
        )
        print(
            f"   {i:<4}{'lambda_2 QR':<26}{value:>12.3e}{abs(value):>12.3e}"
            f"{tol.LORENZ_LAMBDA2_TOL:>10.0e}{_ratio(abs(value), tol.LORENZ_LAMBDA2_TOL):>9}  "
            f"{verdict}"
        )
        print(f"   {i:<4}{'lambda_3 QR (report only)':<26}{float(lambdas[2]):>12.6f}")
    print(
        "   Controls: must fail on the bar (or be not finite, for QR without re-orthonormalization)"
    )
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
    error = abs(value - tol.LORENZ_LAMBDA1_REF)
    verdict = _control(
        failures,
        "D-081 control, Benettin without renormalization",
        {"bar": EXCEEDS},
        chk.check_at_most,
        error,
        bar,
        what="control: Benettin without renormalization",
    )
    name = "Benettin, no renormalization"
    print(
        f"   {0:<4}{name:<26}{value:>12.6f}{error:>12.3e}{bar:>10.5f}{_ratio(error, bar):>9}  "
        f"{verdict}"
    )
    steps_per_qr = lc.lab_value("LORENZ_STEPS_PER_QR")
    result = lc.qr_with(
        lc.lorenz,
        lc.lorenz_initial_conditions()[0],
        lc.LORENZ,
        lc.lorenz_frames()[0],
        dt=tol.LORENZ_DT,
        steps_per_qr=steps_per_qr,
        n_transient=lc.intervals(lc.lorenz_transient_steps(), steps_per_qr),
        n_qr=lc.intervals(window, steps_per_qr),
        orthonormalize=lc.keep_frame,
        step_logdet=False,
    )
    value = float(np.asarray(lyapunov.exponents(result.log_r, steps_per_qr * tol.LORENZ_DT))[1])
    verdict = _control(
        failures,
        "D-081 control, QR without re-orthonormalization",
        {"bound": EXCEEDS, "not finite": NOT_FINITE},
        chk.check_at_most,
        abs(value),
        tol.LORENZ_LAMBDA2_TOL,
        what="control: Lorenz without QR",
    )
    name = "lambda_2, no QR"
    print(
        f"   {0:<4}{name:<26}{value:>12.3e}{abs(value):>12.3e}{tol.LORENZ_LAMBDA2_TOL:>10.0e}"
        f"{_ratio(abs(value), tol.LORENZ_LAMBDA2_TOL):>9}  {verdict}"
    )
    print(f"   section 4 took {_seconds(start)}")


# ---- 5. D-082, D-085 -----------------------------------------------------------------------


def report_agreement(failures: list[str]) -> None:
    """Section 5: Benettin against QR's first exponent, per run, under the worst-case bound."""
    start = time.perf_counter()
    print(
        "\n5. Benettin against QR (D-082, D-085): |lambda_B - lambda_QR,1| <= delta_BQ = "
        f"s eps X / (d0 dt), s = {tol.DELTA_BQ_S:g},\n   valid if X <= X_cap. Report only: "
        f"delta_pr = z delta_BQ / sqrt(N_total), z = {tol.DELTA_BQ_Z:g}, and 'low power' "
        f"where delta_BQ > {tol.LOW_POWER_FRACTION} |lambda_QR,1|"
    )
    print(
        f"   {'run':<12}{'lambda_B':>11}{'lambda_QR,1':>13}{'|diff|':>11}{'X':>10}{'X_cap':>10}"
        f"{'delta_BQ':>11}{'ratio':>9}{'delta_pr':>11}  verdict"
    )
    pendulum_n = lc.steps(tol.LYAP_T, tol.AGREE_DT)
    lorenz_n = lc.steps(tol.LORENZ_T, tol.LORENZ_DT)
    rows = [
        (
            f"pendulum {i}",
            lc.pendulum_run(i),
            lc.pendulum_x_cap(lc.pendulum_states()[i]),
            tol.AGREE_DT,
            pendulum_n,
        )
        for i in range(tol.N_LYAP_PENDULUM)
    ]
    rows += [
        (f"Lorenz {i}", lc.lorenz_run(i), lc.lorenz_x_cap(), tol.LORENZ_DT, lorenz_n)
        for i in range(tol.N_LORENZ_ICS)
    ]
    for name, run, x_cap, dt, n_total in rows:
        lambda_qr = float(run.lambdas_qr[0])
        x = float(run.benettin.max_abs_state)
        delta = lc.delta_bq(x, dt)
        diff = abs(run.lambda_b - lambda_qr)
        verdict = m3._judge(
            failures,
            f"D-082 {name}",
            lc.check_agreement,
            run.lambda_b,
            lambda_qr,
            x,
            x_cap,
            dt,
            what=f"{name} agreement",
        )
        mark = "  low power" if delta > tol.LOW_POWER_FRACTION * abs(lambda_qr) else ""
        print(
            f"   {name:<12}{run.lambda_b:>11.6f}{lambda_qr:>13.6f}{diff:>11.3e}{x:>10.3f}"
            f"{x_cap:>10.1f}{delta:>11.3e}{_ratio(diff, delta):>9}"
            f"{lc.delta_bq_probabilistic(x, dt, n_total):>11.3e}  {verdict}{mark}"
        )
    print(
        "   Control: QR on the (theta, omega) field against Benettin in (q, p); the largest ratio "
        "over the 8 states must exceed 1"
    )
    steps_per_qr = lc.lab_value("PENDULUM_STEPS_PER_QR")
    ratios = []
    for i in range(tol.N_LYAP_PENDULUM):
        run = lc.pendulum_run(i)
        theta = lc.pendulum_theta_qr(i)
        lambda_theta = float(
            np.asarray(lyapunov.exponents(theta.log_r, steps_per_qr * tol.AGREE_DT))[0]
        )
        x = float(run.benettin.max_abs_state)
        x_cap = lc.pendulum_x_cap(lc.pendulum_states()[i])
        m3._judge(
            failures,
            f"D-082 control, X cap at state {i}",
            lc.check_x_cap,
            x,
            x_cap,
            what=f"control: coordinate mismatch, state {i}",
        )
        ratios.append(abs(run.lambda_b - lambda_theta) / lc.delta_bq(x, tol.AGREE_DT))
    worst = max(ratios)
    verdict = _control(
        failures,
        "D-082 control, coordinate mismatch",
        {"bound": EXCEEDS},
        chk.check_at_most,
        worst,
        1.0,
        what="control: coordinate mismatch (bound = 1)",
    )
    cells = ", ".join(f"{r:.3g}" for r in ratios)
    print(f"   ratio per state: {cells}; largest {worst:.3g}  {verdict}")
    print(f"   section 5 took {_seconds(start)}")


# ---- 6. Report only: exponents, pairing and the exponent sums ------------------------------


def _log_volume_lines(
    label: str,
    result: lyapunov.QRResult,
    interval: float,
    horizon: float,
    exact: Callable[[int, float], float],
) -> None:
    """Print ``sum lambda_i(t)`` against the exact log-volume rate at the checkpoints.

    Then the totals over the window: ``sum ln|R_ii| - exact`` and ``sum r = sum log_det_steps -
    exact``.
    """
    log_r = np.asarray(result.log_r)
    cells = []
    for t, count in _checkpoints(horizon, interval):
        value = float(np.sum(log_r[:count])) / t
        cells.append(f"t = {t:g}: {value:+.6e} (minus exact {value - exact(count, t):+.3e})")
    n = log_r.shape[0]
    total_exact = exact(n, horizon) * horizon
    qr_total = float(np.sum(log_r)) - total_exact
    sum_r = float(np.sum(np.asarray(result.log_det_steps))) - total_exact
    print(f"   {label}: " + "; ".join(cells))
    print(f"   {'':<{len(label)}}  sum ln|R_ii| - exact {qr_total:+.3e}, sum r {sum_r:+.3e}")


def report_sums() -> None:
    """Section 6: the exponents, the pairing sums and the exponent-sum convergence (report only)."""
    start = time.perf_counter()
    m1, m2, _, _, _ = lc.DEFAULT
    pendulum_q = lc.lab_value("PENDULUM_STEPS_PER_QR")
    tau_q = pendulum_q * tol.AGREE_DT
    print(
        "\n6. Report only (D-079, D-082): the exponents, the pairing sums and the exponent sums "
        "against the exact\n   log-volume rate, at the powers of ten on a QR boundary; "
        "'sum r' is sum log_det_steps - exact over the window"
    )
    print(
        f"   pendulum (q, p), T = {tol.LYAP_T}: the four exponents, the pairing sums, and times T"
    )
    print(
        f"   {'state':>5}{'lambda_1':>11}{'lambda_2':>11}{'lambda_3':>11}{'lambda_4':>11}"
        f"{'l1 + l4':>12}{'l2 + l3':>12}{'(l1+l4) T':>11}{'(l2+l3) T':>11}"
    )
    for i in range(tol.N_LYAP_PENDULUM):
        lam = lc.pendulum_run(i).lambdas_qr
        a, b = float(lam[0] + lam[3]), float(lam[1] + lam[2])
        print(
            f"   {i:>5}"
            + "".join(f"{v:>11.6f}" for v in lam)
            + f"{a:>12.3e}{b:>12.3e}{a * tol.LYAP_T:>11.3e}{b * tol.LYAP_T:>11.3e}"
        )
    times = ", ".join(f"{t:g}" for t, _ in _checkpoints(tol.LYAP_T, tau_q))
    print(f"   running lambda_1 (q, p) at the checkpoints t = {times}")
    for i in range(tol.N_LYAP_PENDULUM):
        running = np.asarray(lyapunov.running_exponents(lc.pendulum_run(i).qr.log_r, tau_q))[:, 0]
        cells = ", ".join(
            f"{running[count - 1]:.6f}" for _, count in _checkpoints(tol.LYAP_T, tau_q)
        )
        print(f"   state {i}: {cells}")
    envelope = math.log(1.0 + m2 / m1)
    print(
        f"   pendulum (theta, omega): exact rate (1/t) ln[det M(theta_0) / det M(theta_t)], "
        f"inside the envelope ln(1 + m2/m1) / t = {envelope:.4f} / t"
    )
    for i in range(tol.N_LYAP_PENDULUM):
        result = lc.pendulum_theta_qr(i)
        states = np.asarray(result.states)
        first = float(lc._log_det_m(jnp.asarray(states[0, :2]), lc.DEFAULT))

        def d008(count: int, t: float, states: np.ndarray = states, first: float = first) -> float:
            return (first - float(lc._log_det_m(jnp.asarray(states[count, :2]), lc.DEFAULT))) / t

        _log_volume_lines(f"state {i}", result, tau_q, tol.LYAP_T, d008)
    print("   pendulum (q, p): exact rate 0")
    for i in range(tol.N_LYAP_PENDULUM):
        _log_volume_lines(f"state {i}", lc.pendulum_run(i).qr, tau_q, tol.LYAP_T, lambda c, t: 0.0)
    lorenz_tau = lc.lab_value("LORENZ_STEPS_PER_QR") * tol.LORENZ_DT
    rate = -(tol.LORENZ_SIGMA + 1.0 + tol.LORENZ_BETA)
    print(f"   Lorenz: exact rate -(sigma + 1 + beta) = {rate:.6f}")
    for i in range(tol.N_LORENZ_ICS):
        _log_volume_lines(
            f"ic {i}", lc.lorenz_run(i).qr, lorenz_tau, tol.LORENZ_T, lambda c, t: rate
        )
    print(f"   section 6 took {_seconds(start)}")


def main() -> int:
    """Print every section and return the exit status."""
    failures: list[str] = []
    m3.print_provenance()
    print(
        "\nMilestone 5, the Lyapunov estimators: the gates of D-079 to D-082 (D-085, D-086), "
        "with the lab's values of D-091"
    )
    print(
        f"   pendulum tau_Q = {lc.lab_value('PENDULUM_STEPS_PER_QR')} steps, tau_B = "
        f"{lc.lab_value('PENDULUM_STEPS_PER_RENORM')} steps, dt = {tol.AGREE_DT}, t_trans = "
        f"{tol.PENDULUM_T_TRANS}, T = {tol.LYAP_T}, d0 = {tol.LYAP_D0:g}"
    )
    report_identity(failures)
    report_step(failures)
    report_linear(failures)
    report_lorenz(failures)
    report_agreement(failures)
    report_sums()
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED. Nothing was adjusted (D-025):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All checks are within the declared thresholds, and every control fails as required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
