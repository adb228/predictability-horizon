# Progress

Running log: what is done, what is next, open questions. Updated at the end of every session (brief §8). The approved Stage 1 plan, with its seven milestones and their gates, is in `docs/plans/stage1.md`.

## Where we are (2026-10-11)

Stage 1, milestone 1 of 7 (repository baseline): **done.** The gate passed locally and in CI on the private remote. Since then the project requires Python 3.12 or newer, with one locked stack and one pinned CI job (D-026, D-027). No scientific results exist yet.

Milestone 2 of 7 (double-pendulum mechanics): **done and merged** (pull request #1, merge commit `5373671`, 2026-10-11; the branch is deleted). Design reviewed by Opus 5.5 and logged before any code (D-028 to D-034). It adds the derivation note, the frozen tolerances with the single comparison rule, an independent SymPy/mpmath oracle with its self-checks, the tests T1 to T11, and the six functions in `src/horizon/systems/double_pendulum.py`, implemented by the agent from the equations the owner derived and verified (D-039, which now covers all implementation). The first Linux CI run, before the functions existed, was red only where expected (D-038). CI is green on the pull request head (409 tests in 35 s) and again on `main` after the merge (409 tests in 40 s), both on Linux (D-040). The agent's review of the six functions against the derivation note found no behavioural issue and three docstring nits (queued below); the reviewer is also the author, so this is a self-review.

Milestone 3 of 7 (integrators): **approved by the owner on 2026-10-11 (D-041); design reviewed by Opus 5.5 and logged before any code (D-042 to D-049); no code and no test yet** (see Next). The gates are predeclared: an exact-solution gate on four problems, a pendulum convergence gate, an agreement gate (chaotic-ish states: `dt = 0.01` and horizon `T = 8` in units of `√(L1/g)`, tolerance 1e-4; regular states: `T = 50`, tolerance 1e-6), and a wiring check with an energy anchor. The calibration behind them is uncommitted lab work, not results.

## Done

- Calibration, and the Stage 1 plan approved with amendments (D-001 to D-012).
- Milestone 1 decisions reviewed by Opus 5.5 before implementation (D-013 to D-020, D-024). Owner confirmations recorded (D-021 to D-023).
- Repository baseline: brief and decision log, project scaffold, tests, CI workflow, this log.
- Local gate for milestone 1, run on both locked stacks that existed at the time (Python 3.11 support was removed afterwards, D-027), with the same commands CI uses:

  | Python | jax | numpy | scipy | `ruff check` | `ruff format --check` | `pytest` |
  |---|---|---|---|---|---|---|
  | 3.12 (reference) | 0.11.2 | 2.5.3 | 1.18.1 | pass | pass | 5 passed |
  | 3.11 | 0.10.2 | 2.4.6 | 1.17.1 | pass | pass | 5 passed |

- Two mutation checks showed that the float64 test and the fatal-warning test can fail (D-015).
- Private GitHub remote `adb228/predictability-horizon` (D-023). The first push needed the `workflow` scope, which the owner granted to the `gh` token. CI run 38039152732 passed on Python 3.12 (19 s) and 3.11 (21 s).
- Milestone 2, pull request #1 (merged, D-040; its first Linux CI run, 38100606420, was red as intended, D-038): `docs/derivations/double_pendulum.md`; `src/horizon/tolerances.py` with `check_close` and its tests; `tests/oracles/double_pendulum.py` and `tests/cases.py` with the self-checks O1 to O7 (`sympy` and `mpmath` added as dev dependencies, D-014); `src/horizon/systems/double_pendulum.py` as an interface skeleton; `tests/test_double_pendulum.py` with T1 to T11. Uncommitted labs run before any owner code exists (D-033 stage a) showed that the oracle's self-checks and the tests can fail on deliberately wrong code and pass on correct code; they are not results and their scripts are not committed.
- Milestone 2, after the owner's questions: a per-test timeout of 60 s (`pytest-timeout`, D-036) with a guard test, `tests/test_timeout_config.py`, so that a hung implementation fails instead of blocking the run; D-035 records the confirmations that the oracle catches a wrong coupling angle and that the residual tests are numerical, not symbolic.
- Milestone 2: the six mechanics functions, implemented by the agent from the owner's derivation (D-039). `tests/test_double_pendulum.py` and the whole suite pass locally and Ruff is clean; no test, tolerance or oracle file changed. The `/tmp` lab is deleted. D-037 is trimmed to one line at the owner's request.
- Authorship rule broadened the same day (D-039): the owner derives and verifies, and the agent implements all code. Before implementing the Lyapunov estimators (milestone 5), the agent walks the owner through the math. The wording fix for "the owner's code" is queued below.
- Milestone 2 merged and closed (D-040): merge commit `5373671` (no squash, D-033.3), branch deleted, CI green on the pull request and on `main`. The three interview questions (brief §7) were given on 2026-10-10. There is no mutation script (D-038).
- Milestone 3 design reviewed by Opus 5.5 and logged before any code (D-042 to D-049): the interface of `horizon.integrators`, the RK4 scheme, four exact-solution problems (two of them nonlinear), the convergence and agreement gates, a 20-sample tail check that moved the chaotic agreement horizon from `T = 10` to `T = 8` under Opus 5.5's own rule, a wiring check with an energy anchor, and the register and script. The lab behind it (`/tmp/m3lab`) is uncommitted and not results; it is deleted when milestone 3 closes.

## Next

- **Milestone 3 (integrators), approved by the owner (D-041), design logged (D-042 to D-049), not started.** Order of work, as in milestone 2: (1) the derivation note `docs/derivations/integrators.md` with its claim-to-test table; (2) the tests, with every threshold in `src/horizon/tolerances.py`, and an interface skeleton, red; (3) `src/horizon/integrators.py`, green, with `scipy` declared as a direct dependency in the commit that first imports it (D-014); (4) `experiments/stage1/integrator_validation.py`, which prints the measured numbers with a provenance header and writes no files; then a pull request, Linux CI and a merge commit. Any change to a logged threshold, grid or horizon goes to Opus 5.5 first (D-025). Pushing the decision-log commit and opening the branch `milestone-3-integrators` wait for the owner's go-ahead, since pushing is the owner's decision. The physics validation tests (energy drift at `t = 1000`, normal-mode FFT, time reversibility) belong to milestone 4. Three interview questions at the end of the milestone (brief §7).
- Milestone 3, before any code: the short brief §7 walkthrough, with no derive-and-verify step (the owner's choice, D-041): why RK4's global error is `O(dt⁴)`, how DOP853's embedded error estimate controls the step size, and why two integrators can agree only up to a finite horizon in a chaotic system. Given in chat on 2026-10-11, before any milestone 3 code.
- Carried forward from the milestone 3 review: the DOP853 reference is only a pointwise reference over limited horizons in chaotic motion (D-046 and the risk list after D-049), and float64 conditioning of unwrapped angles degrades with `|θ|` (D-032). Both bear on milestones 5 and 6 and on Stage 2 valid-prediction time.
- Milestone 5 (Lyapunov estimators), before any code: the agent walks the owner through the math; the agent implements only after the owner has derived and verified it (D-039, the owner's instruction).

## Queued for the pre-public docs cleanup

One docs-only pass before the repository goes public (after the milestone 4 gate, D-012). The owner asked for the first item to be queued (D-039); the second is the agent's suggestion. Nothing here is done yet.

- **Wording "the owner's code".** The code under test is written by the agent from the owner's derivation (D-039), but 17 places still call it "the owner's code", "owner-written" or "your code", and some also use the old tense ("they stay red until the owner implements", "before the owner writes any code"): `docs/derivations/double_pendulum.md` (3: the section 9 intro and the T4 and O7 rows), `tests/test_double_pendulum.py` (4), `tests/test_oracle_double_pendulum.py` (9) and one comment in `src/horizon/tolerances.py` (1). The decision log keeps its historical wording. That pass changes comments and docstrings only: no test logic and no tolerance value (D-025).
- **Docstring nits in `src/horizon/systems/double_pendulum.py`** (from the agent's self-review; behaviour unchanged; strike any you disagree with): the `mass_matrix` docstring points to section 3 of the note, but `M` is defined in section 2; `Params` does not say that `m1`, `m2`, `L1` and `L2` must be positive (the note's section 2 says it for the masses only), although `acceleration` divides by `det M`, which is zero if `m2`, `L1` or `L2` is zero, or if `m1 = 0` and `sin Δ = 0`; `acceleration` evaluates `sin Δ` twice.

## Open questions

- For the owner to decide before the repository goes public: the tool that runs the agent's git commands adds a `Co-authored-by: Cursor <cursoragent@cursor.com>` line to commit messages (28 of the 30 commits on `main` so far; the exceptions are `27434b0`, where it was not added, and the merge commit `5373671`, which GitHub made). Removing it from existing commits would mean rewriting history. See D-039, Attribution.
- For the owner's information: `docs/BRIEF.md` §4 still says "Python 3.11+". D-027 records the deviation; edit the brief if you want it to match.
