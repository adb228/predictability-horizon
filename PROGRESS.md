# Progress

Running log: what is done, what is next, open questions. Updated at the end of every session (brief §8).

## Where we are (2026-10-11)

Stage 1, milestone 1 of 7 (repository baseline): **done.** The gate passed locally and in CI on the private remote. Since then the project requires Python 3.12 or newer, with one locked stack and one pinned CI job (D-026, D-027). No scientific results exist yet.

Milestone 2 of 7 (double-pendulum mechanics): **done and merged** (pull request #1, merge commit `5373671`, 2026-10-11; the branch is deleted). Design reviewed by Opus 5.5 and logged before any code (D-028 to D-034). It adds the derivation note, the frozen tolerances with the single comparison rule, an independent SymPy/mpmath oracle with its self-checks, the tests T1 to T11, and the six functions in `src/horizon/systems/double_pendulum.py`, implemented by the agent from the equations the owner derived and verified (D-039, which now covers all implementation). The first Linux CI run, before the functions existed, was red only where expected (D-038). CI is green on the pull request head (409 tests in 35 s) and again on `main` after the merge (409 tests in 40 s), both on Linux (D-040). The agent's review of the six functions against the derivation note found no behavioural issue and three docstring nits (queued below); the reviewer is also the author, so this is a self-review.

Milestone 3 of 7 (integrators): **proposed, not started; waiting for the owner's decisions** (see Next and Open questions).

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

## Next

- **Milestone 3 (integrators), proposed, not started.** `src/horizon/integrators.py` with a fixed-step RK4 in JAX (`lax.scan`, batched over initial conditions with `vmap`) and a SciPy DOP853 reference wrapper (`rtol = atol = 1e-12`, brief §3). `scipy` becomes a direct dependency in the commit that first imports it (D-014). Gate, as in the plan: RK4 converges at fourth order against the reference, and RK4 agrees with DOP853 to a stated tolerance over a stated horizon. The public interface, the thresholds, the `dt` grid and the horizon go to Opus 5.5 and are logged before any code (D-007, D-025). The physics validation tests (energy drift at `t = 1000`, normal-mode FFT, time reversibility) belong to milestone 4.
- Milestone 3, before any code: the brief §7 walkthrough of the integrator math, and the owner's decision whether to derive and verify it too (D-039).
- Milestone 5 (Lyapunov estimators), before any code: the agent walks the owner through the math; the agent implements only after the owner has derived and verified it (D-039, the owner's instruction).

## Queued for the pre-public docs cleanup

One docs-only pass before the repository goes public (after the milestone 4 gate, D-012). The owner asked for the first item to be queued (D-039); the second is the agent's suggestion. Nothing here is done yet.

- **Wording "the owner's code".** The code under test is written by the agent from the owner's derivation (D-039), but 17 places still call it "the owner's code", "owner-written" or "your code", and some also use the old tense ("they stay red until the owner implements", "before the owner writes any code"): `docs/derivations/double_pendulum.md` (3: the section 9 intro and the T4 and O7 rows), `tests/test_double_pendulum.py` (4), `tests/test_oracle_double_pendulum.py` (9) and one comment in `src/horizon/tolerances.py` (1). The decision log keeps its historical wording. That pass changes comments and docstrings only: no test logic and no tolerance value (D-025).
- **Docstring nits in `src/horizon/systems/double_pendulum.py`** (from the agent's self-review; behaviour unchanged; strike any you disagree with): the `mass_matrix` docstring points to section 3 of the note, but `M` is defined in section 2; `Params` does not say that `m1`, `m2`, `L1` and `L2` must be positive (the note's section 2 says it for the masses only), although `acceleration` divides by `det M`, which is zero if `m2`, `L1` or `L2` is zero, or if `m1 = 0` and `sin Δ = 0`; `acceleration` evaluates `sin Δ` twice.

## Open questions

- Milestone 3 needs two answers from the owner before any code: (1) derive and verify the integrator math yourself, or is the brief §7 walkthrough enough? (D-039 says the agent asks at this milestone.) (2) Does the scope under Next match your plan?
- The seven-milestone Stage 1 list is not written down in the repository; only pointers in the decision log exist. The agent's reconstruction, for the owner to confirm before it becomes a section of this file: 1 repository baseline; 2 double-pendulum mechanics; 3 integrators; 4 physics validation gate (the repository goes public after it, D-012); 5 Lyapunov estimators; 6 sensitivity and regime mapping (D-011); 7 figures, report and release.
- For the owner to decide before the repository goes public: the tool that runs the agent's git commands adds a `Co-authored-by: Cursor <cursoragent@cursor.com>` line to commit messages (26 of the 28 commits on `main` so far; the exceptions are `27434b0`, where it was not added, and the merge commit `5373671`, which GitHub made). Removing it from existing commits would mean rewriting history. See D-039, Attribution.
- For the owner's information: `docs/BRIEF.md` §4 still says "Python 3.11+". D-027 records the deviation; edit the brief if you want it to match.
