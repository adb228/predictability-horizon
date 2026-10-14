"""Print the milestone 4 physics validation: the brief's five validation tests and their numbers.

Run from the repository root::

    uv run python -m experiments.stage1.physics_validation

It writes no files and makes no figures (D-052). Tests 1 to 3 of the brief (energy drift, the
small-angle FFT, time reversibility) are new in milestone 4; tests 4 and 5 (convergence order,
agreement) are milestone 3's, printed by milestone 3's own report functions so that all five
appear in one printout. It uses the same measurement and check code as the tests
(``tests/physics_checks.py``, ``tests/integrator_checks.py``) and the declared values of
``horizon.tolerances``, so each number printed here is the number its gate judged, printed next
to its threshold. Nothing is chosen here. The exit status is 1 if any check fails and 0
otherwise.

A number printed by this script is a measurement made at the commit named in the provenance
header, which must say "clean" for the printout to be pasted into the decision log (D-060). Any
milestone 4 number that appears later in a report must come from the milestone 7 JSON entry
point and not from copied printout (D-048).
"""

import sys
import time

import numpy as np

from experiments.stage1 import integrator_validation as m3
from horizon import tolerances as tol
from horizon.tolerances import check_close
from tests import integrator_checks as chk
from tests import physics_checks as ph

P = ph.DEFAULT


def _seconds(start: float) -> str:
    return f"({time.perf_counter() - start:.0f} s)"


def _drift_table(
    failures: list[str] | None, label: str, states: np.ndarray, *, gated: bool
) -> list[ph.DriftMeasurement]:
    """Print one DOP853 energy-drift row per state, and return the measurements."""
    scale = chk.energy_scale(P)
    ceiling = tol.E_DOP_CEILING
    growth = "".join(f"{'D(' + str(t) + ')':>11}" for t in tol.E_GROWTH_TIMES)
    print(f"   {label}")
    print(f"   {'state':>5}{'E(0)':>9}{growth}{'D/ceiling':>11}{'literal':>13}  verdict")
    measurements = []
    for i, state in enumerate(states):
        m = ph.measure_drift(ph.dop853_energies(state), P)
        measurements.append(m)
        if gated:
            assert failures is not None
            verdict = m3._judge(
                failures, f"E1 {label} state {i}", chk.check_at_most, m.drift, ceiling, what="drift"
            )
        else:
            verdict = "report only"
        literal = f"{m.literal:>13.2e}" if m.literal is not None else f"{'below floor':>13}"
        cells = "".join(f"{g:>11.2e}" for g in m.growth)
        print(f"   {i:>5}{m.e0:>9.3f}{cells}{m.drift / ceiling:>11.3f}{literal}  {verdict}")
    worst, median, smallest = ph.summary(np.array([m.drift for m in measurements]))
    print(
        f"   drift over E_s: maximum {worst:.3e}, median {median:.3e}, minimum {smallest:.3e}; "
        f"the ceiling is {ceiling / worst:.2f} times the maximum"
    )
    shown = [m for m in measurements if m.literal is not None]
    above = [m.literal for m in shown if m.literal is not None and m.literal > ceiling]
    print(
        f"   literal |dE| / |E(0)| shown for {len(shown)} of {len(measurements)} states "
        f"(|E(0)| >= {tol.E_LITERAL_FLOOR * scale:.3g}); {len(above)} of them exceed {ceiling:.0e}"
        + (f" (largest {max(above):.2e})" if above else "")
    )
    return measurements


def report_energy(failures: list[str]) -> None:
    """Section 1: brief test 1, energy drift (D-053 to D-055)."""
    start = time.perf_counter()
    scale = chk.energy_scale(P)
    print(
        f"\n1. Energy drift (D-053 to D-055): max over integer times of |E(t) - E(0)| / E_s, "
        f"E_s = {scale:g}"
    )
    print(
        f"   E1: DOP853 at its defaults (rtol = atol = 1e-12), T = {tol.E_HORIZON}; ceiling "
        f"{tol.E_DOP_CEILING:.0e}, the brief's number."
    )
    print("   D(t) is the running maximum up to t (the growth law); 'literal' is |dE| / |E(0)|.")
    _drift_table(failures, "regular", ph.e1_regular_states(), gated=True)
    _drift_table(failures, "chaotic-ish", ph.e1_chaotic_states(), gated=True)
    moving = _drift_table(None, "moving (not gated, D-054)", ph.moving_states(), gated=False)
    closest = max(m.drift for m in moving)
    print(
        f"   moving states come within {tol.E_DOP_CEILING / closest:.2f} times of the ceiling "
        "(the margin of the reference tolerance, which is why they are not gated)."
    )

    print(f"\n   E2: RK4 at dt = {tol.AGREE_DT}, T = {tol.E_HORIZON}")
    for label, states, ceiling in (
        ("regular", ph.e2_regular_states(), tol.E2_CEILING_REGULAR),
        ("chaotic-ish", ph.e2_chaotic_states(), tol.E2_CEILING_CHAOTIC),
    ):
        values = ph.drift(ph.rk4_energies(states, P), P)
        worst, median, smallest = ph.summary(values)
        verdict = m3._judge(
            failures, f"E2 {label}", chk.check_at_most, worst, ceiling, what="RK4 drift"
        )
        print(
            f"   {label:<12} n = {len(states):>2}  maximum {worst:.3e}  median {median:.3e}  "
            f"minimum {smallest:.3e}  ceiling {ceiling:.0e}  ratio {worst / ceiling:.4f}  "
            f"{verdict}"
        )
    values = ph.drift(ph.rk4_energies(ph.moving_states(), P), P)
    worst, median, smallest = ph.summary(values)
    print(
        f"   {'moving':<12} n = {len(values):>2}  maximum {worst:.3e}  median {median:.3e}  "
        f"minimum {smallest:.3e}  report only"
    )
    print(f"   section 1 took {_seconds(start)}")


def report_fft(failures: list[str]) -> None:
    """Section 2: brief test 2, the small-angle FFT (D-056)."""
    start = time.perf_counter()
    print(
        f"\n2. Small-angle FFT (D-056): A = {tol.F_AMPLITUDE:g}, {tol.F_N_SAMPLES} samples at "
        f"{tol.F_SAMPLE_DT} t_s, tolerance {tol.F_TOL:.0e} relative"
    )
    print(
        f"   {'set':<4}{'integrator':<11}{'mode':<9}{'peak':>5}{'measured':>12}{'analytic':>12}"
        f"{'rel. error':>12}{'ratio':>8}  verdict"
    )
    for set_name, params in ph.F_SETS.items():
        for integrator in ("dop853", "rk4"):
            for mode in ph.MODE_DIRECTIONS:
                measured = ph.fft_peaks(params, mode, integrator)
                expected = ph.expected_peaks(mode, params)
                errors = ph.relative_errors(measured, expected)
                verdict = m3._judge(
                    failures,
                    f"F {set_name} {integrator} {mode}",
                    chk.check_at_most,
                    max(errors),
                    tol.F_TOL,
                    what="peak error",
                )
                for k, (m, e, err) in enumerate(zip(measured, expected, errors, strict=True)):
                    print(
                        f"   {set_name:<4}{integrator:<11}{mode:<9}{k + 1:>5}{m:>12.7f}"
                        f"{e:>12.7f}{err:>12.2e}{err / tol.F_TOL:>8.3f}  {verdict}"
                    )
    print(f"   section 2 took {_seconds(start)}")


def report_round_trip(failures: list[str]) -> None:
    """Section 3: brief test 3, time reversibility (D-057)."""
    start = time.perf_counter()
    print(
        "\n3. Time reversibility (D-057): forward T, flip omega, forward T, flip back; the largest "
        "absolute\n   difference from the initial state over the four components"
    )
    for label, states, horizon, ceilings in (
        (
            "regular",
            ph.v_regular_states(),
            tol.V_REGULAR_T,
            (tol.V1_CEILING_REGULAR, tol.V2_CEILING_REGULAR),
        ),
        (
            "chaotic-ish",
            ph.v_chaotic_states(),
            tol.V_CHAOTIC_T,
            (tol.V1_CEILING_CHAOTIC, tol.V2_CEILING_CHAOTIC),
        ),
    ):
        v1 = np.array([ph.dop853_round_trip_error(s, horizon) for s in states])
        v2 = ph.rk4_round_trip_errors(states, horizon)
        for name, values, ceiling in (
            ("V1 DOP853", v1, ceilings[0]),
            (f"V2 RK4 dt = {tol.AGREE_DT}", v2, ceilings[1]),
        ):
            worst, median, smallest = ph.summary(values)
            verdict = m3._judge(
                failures, f"{name} {label}", chk.check_at_most, worst, ceiling, what="round trip"
            )
            print(
                f"   {name:<17}{label:<12} T = {horizon:<4} n = {len(states):>2}  maximum "
                f"{worst:.3e}  median {median:.3e}  minimum {smallest:.3e}  ceiling "
                f"{ceiling:.0e}  ratio {worst / ceiling:.4f}  {verdict}"
            )
    worst_difference = 0.0
    verdicts = []
    for label, states in (
        ("regular", ph.v_regular_states()),
        ("chaotic-ish", ph.v_chaotic_states()),
    ):
        negative, flipped = ph.rk4_negative_dt_pair(states, tol.V_CHAOTIC_T)
        worst_difference = max(worst_difference, chk.max_abs_diff(negative, flipped))
        verdicts.append(
            m3._judge(
                failures,
                f"V3 {label}",
                check_close,
                negative,
                flipped,
                rtol=tol.IDENTITY_RTOL,
                atol=tol.IDENTITY_ATOL,
                what="negative dt against the flipped forward step",
            )
        )
    print(
        f"   V3 RK4 with -dt against S RK4(dt) S, T = {tol.V_CHAOTIC_T}: largest difference "
        f"{worst_difference:.2e}  {'/'.join(verdicts)} (regular/chaotic-ish)"
    )
    print(f"   section 3 took {_seconds(start)}")


def main() -> int:
    """Print every section and return the exit status."""
    failures: list[str] = []
    m3.print_provenance()
    print("\nThe brief's five validation tests (docs/BRIEF.md section 3, Stage 1)")
    report_energy(failures)
    report_fft(failures)
    report_round_trip(failures)
    print("\n4. Convergence order (brief test 4): milestone 3's G1 and G2, sections A and B below")
    m3.report_exact(failures)
    m3.report_convergence(failures)
    print(
        "\n5. Agreement of RK4 and DOP853 (brief test 5): milestone 3's G3 and G4, section C below"
    )
    m3.report_agreement(failures)
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED. Nothing was adjusted (D-025):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All checks are within the declared thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
