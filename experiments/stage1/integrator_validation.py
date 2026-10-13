"""Print the milestone 3 integrator validation: slopes, errors, margins and the growth curve.

Run from the repository root::

    uv run python -m experiments.stage1.integrator_validation

It writes no files and makes no figures (D-048). It uses the same measurement and check code as
the tests (``tests/integrator_checks.py``) and the declared values of ``horizon.tolerances``, so
each number printed here is the number its gate judged, printed next to its threshold. Nothing is
chosen here. The exit status is 1 if any check fails and 0 otherwise.

A number printed by this script is a measurement made at the commit named in the provenance
header. Any milestone 3 number that appears later in a report must come from the milestone 7
JSON entry point and not from copied printout (D-048).
"""

import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Callable

import numpy as np

from horizon import tolerances as tol
from tests import integrator_checks as chk


def _git(*args: str) -> str:
    """Return the output of a git command run in the current directory, or "" on failure."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def print_provenance() -> None:
    """Print the git commit, whether the working tree is dirty, and the package versions."""
    commit = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain"))
    print("Provenance (D-019, D-048)")
    print(f"  git commit   {commit}" + ("   (working tree is DIRTY)" if dirty else "   (clean)"))
    print(f"  python       {platform.python_version()}")
    for package in ("jax", "jaxlib", "numpy", "scipy"):
        print(f"  {package:<12} {importlib.metadata.version(package)}")


def _judge(
    failures: list[str], label: str, check: Callable[..., float], *args: object, **kwargs: object
) -> str:
    """Run ``check(*args, **kwargs)``; return ``ok``, or ``FAIL`` and record it under ``label``."""
    try:
        check(*args, **kwargs)
    except AssertionError as error:
        failures.append(f"{label}: {error}")
        return "FAIL"
    return "ok"


def report_exact(failures: list[str]) -> None:
    """Section A: RK4 and the wrapper against the four closed-form solutions (D-044)."""
    low, high = tol.EXACT_SLOPE_BAND
    finest = int(np.argmin(tol.EXACT_DT_GRID))
    print("\nA. Exact-solution problems (D-044): worst |RK4 - exact| over t = 0, 1, ..., T")
    print(f"   grid dt = {', '.join(str(dt) for dt in tol.EXACT_DT_GRID)}")
    print(f"   slope band [{low}, {high}]; wrapper bound {tol.EXACT_WRAPPER_TOL:.0e}")
    columns = "".join(f"{'err(' + str(dt) + ')':>13}" for dt in tol.EXACT_DT_GRID)
    print(f"   {'problem':<10}{'T':>3}{columns}{'slope':>8}{'finest/ceiling':>16}{'wrapper':>11}")
    for problem in chk.exact_problems():
        errors = chk.rk4_exact_errors(problem)
        slope = chk.log_log_slope(tol.EXACT_DT_GRID, errors)
        wrapper = chk.wrapper_exact_error(problem)
        name = problem.name
        verdict = "/".join(
            (
                _judge(
                    failures,
                    f"{name}: slope",
                    chk.check_in_band,
                    slope,
                    tol.EXACT_SLOPE_BAND,
                    what="slope",
                ),
                _judge(
                    failures,
                    f"{name}: ceiling",
                    chk.check_at_most,
                    float(errors[finest]),
                    problem.ceiling,
                    what="finest-step error",
                ),
                _judge(
                    failures,
                    f"{name}: wrapper",
                    chk.check_at_most,
                    wrapper,
                    tol.EXACT_WRAPPER_TOL,
                    what="wrapper error",
                ),
            )
        )
        cells = "".join(f"{e:>13.3e}" for e in errors)
        print(
            f"   {name:<10}{problem.horizon:>3}{cells}{slope:>8.3f}"
            f"{errors[finest] / problem.ceiling:>16.3f}{wrapper:>11.2e}   {verdict}"
        )
    print("   verdicts: slope / ceiling / wrapper. A ratio of at most 1 is inside the bound.")


def report_convergence(failures: list[str]) -> None:
    """Section B: convergence on the regular pendulum states and the validity rule (D-045)."""
    low, high = tol.CONV_SLOPE_BAND
    result = chk.measure_convergence()
    finest = int(np.argmin(tol.CONV_DT_GRID))
    print("\nB. Pendulum convergence (D-045): regular states, default parameters, T = 10")
    print(
        f"   reference rtol = atol = {tol.CONV_REFERENCE_TOL:g}; self-check reference "
        f"{tol.CONV_SELF_CHECK_TOL:g}; slope band [{low}, {high}]"
    )
    print(
        f"   validity: the error at dt = {tol.CONV_DT_GRID[finest]} must be at least "
        f"{tol.CONV_FLOOR_RATIO:g} times the reference's own error"
    )
    print(f"   {'state':>5}{'slope':>8}{'err(finest)':>13}{'self-error':>12}{'ratio':>9}  verdict")
    for i, slope in enumerate(result.slopes):
        error, own = float(result.errors[i, finest]), float(result.self_errors[i])
        verdict = "/".join(
            (
                _judge(
                    failures,
                    f"convergence state {i}: slope",
                    chk.check_in_band,
                    slope,
                    tol.CONV_SLOPE_BAND,
                    what="slope",
                ),
                _judge(
                    failures,
                    f"convergence state {i}: validity",
                    chk.check_at_least,
                    error,
                    tol.CONV_FLOOR_RATIO * own,
                    what="validity rule",
                ),
            )
        )
        print(f"   {i:>5}{slope:>8.3f}{error:>13.3e}{own:>12.3e}{error / own:>9.1f}  {verdict}")
    smallest = float(np.min(result.errors[:, finest] / result.self_errors))
    print(
        f"   slopes {result.slopes.min():.3f} to {result.slopes.max():.3f}; "
        f"smallest error / self-error {smallest:.1f}"
    )
    print("   verdicts: slope / validity.")


def report_agreement(failures: list[str]) -> None:
    """Section C: RK4 against DOP853 on chaotic-ish and regular states (D-046)."""
    chaotic = chk.chaotic_agreement_errors()
    regular = chk.regular_agreement_errors()
    print(f"\nC. Agreement of RK4 and DOP853 (D-046): default parameters, dt = {tol.AGREE_DT}")
    for label, errors, bound, horizon in (
        ("chaotic-ish states", chaotic, tol.CHAOTIC_TOL, tol.CHAOTIC_T),
        ("regular states", regular, tol.REGULAR_AGREE_TOL, tol.REGULAR_AGREE_T),
    ):
        worst = float(np.max(errors))
        verdict = _judge(
            failures, f"agreement, {label}", chk.check_at_most, worst, bound, what=label
        )
        print(
            f"   {label:<19} T = {horizon:<3} worst |RK4 - DOP853| = {worst:.3e}  "
            f"bound {bound:.0e}  ratio {worst / bound:.3f}  {verdict}"
        )
    print("\n   Error growth for the chaotic-ish ensemble (non-gating, D-046(c)):")
    print(f"   {'t':>3}{'median':>12}{'maximum':>12}")
    for t in range(chaotic.shape[1]):
        print(f"   {t:>3}{np.median(chaotic[:, t]):>12.3e}{np.max(chaotic[:, t]):>12.3e}")
    print("\n   Error growth for the regular states (non-gating), maximum over states:")
    print("   " + "  ".join(f"t={t}: {np.max(regular[:, t]):.2e}" for t in range(0, 51, 10)))


def report_wiring(failures: list[str]) -> None:
    """Section D: non-default parameters, the wiring check and the energy anchor (D-047)."""
    measured = chk.measure_all_wiring()
    print(
        f"\nD. Non-default parameters (D-047): {tol.N_WIRE_SETS} sets x {tol.N_WIRE_STATES} "
        f"states, dt = {tol.WIRE_DT} t_s, T = {tol.WIRE_T:g} t_s"
    )
    for label, worst, bound in (
        ("agreement |RK4 - wrapper|", max(m.agreement for m in measured), tol.WIRE_AGREE_TOL),
        ("RK4 energy drift / E_s", max(m.energy_rk4 for m in measured), tol.WIRE_ENERGY_TOL),
        (
            "wrapper energy drift / E_s",
            max(m.energy_wrapper for m in measured),
            tol.WIRE_ENERGY_TOL,
        ),
    ):
        verdict = _judge(failures, f"wiring, {label}", chk.check_at_most, worst, bound, what=label)
        print(
            f"   {label:<28} worst over {len(measured)} cases = {worst:.3e}  "
            f"bound {bound:.0e}  ratio {worst / bound:.4f}  {verdict}"
        )


def main() -> int:
    """Print every section and return the exit status."""
    failures: list[str] = []
    print_provenance()
    report_exact(failures)
    report_convergence(failures)
    report_agreement(failures)
    report_wiring(failures)
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
