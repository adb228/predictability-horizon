# Progress

Running log: what is done, what is next, open questions. Updated at the end of every session (brief §8).

## Where we are (2026-10-10)

Stage 1, milestone 1 of 7 (repository baseline): **done.** The gate passed locally and in CI on the private remote. Since then the project requires Python 3.12 or newer, with one locked stack and one pinned CI job (D-026, D-027). No scientific results exist yet.

Milestone 2 of 7 (double-pendulum mechanics): **in progress, waiting for the owner.** Design reviewed by Opus 5.5 and logged before any code (D-028 to D-034). On branch `milestone-2-mechanics` everything except the owner's code exists: derivation note, frozen tolerances with the single comparison rule, an independent SymPy/mpmath oracle with its self-checks (green), and the interface skeleton with the tests T1 to T11 (red until the owner implements the six functions). The branch is pushed to the private remote, but a branch push starts no CI run (CI runs on pushes to `main` and on pull requests only), so the first Linux run comes with a pull request (D-037). **Your turn:** write the bodies in `src/horizon/systems/double_pendulum.py`.

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
- Milestone 2, on branch `milestone-2-mechanics` (not merged): `docs/derivations/double_pendulum.md`; `src/horizon/tolerances.py` with `check_close` and its tests; `tests/oracles/double_pendulum.py` and `tests/cases.py` with the self-checks O1 to O7 (`sympy` and `mpmath` added as dev dependencies, D-014); `src/horizon/systems/double_pendulum.py` as an interface skeleton; `tests/test_double_pendulum.py` with T1 to T11. Uncommitted labs run before any owner code exists (D-033 stage a) showed that the oracle's self-checks and the tests can fail on deliberately wrong code and pass on correct code; they are not results and their scripts are not committed.
- Milestone 2, after the owner's questions: a per-test timeout of 60 s (`pytest-timeout`, D-036) with a guard test, `tests/test_timeout_config.py`, so that a hung implementation fails instead of blocking the run; D-035 records the confirmations that the oracle catches a wrong coupling angle and that the residual tests are numerical, not symbolic.

## Next

- Milestone 2, owner: implement `mass_matrix`, `kinetic_energy`, `potential_energy`, `energy`, `acceleration` and `vector_field` (D-003), running the frozen tests as you go (for example `uv run pytest tests/test_double_pendulum.py -k "t1_"`; the trailing underscore matters, because `-k t1` is a substring match and also selects T10 and T11). A test that hangs now fails with "Timeout" after 60 s instead of blocking the run (D-036); add `-x` to stop at the first failure, and after any timeout rerun the timed-out test alone before trusting later failures in that session. The implementer reviews and never edits that file; any change to a test or a tolerance goes through D-025.
- Milestone 2, after the owner's code is green: review comments only; then open the pull request (it is also the first Linux CI run), merge with a merge commit, and give the three interview questions. **On hold by the owner's instruction (no more test-infrastructure work this milestone, D-037):** the committed mutation script `scripts/mutation_check.py` of D-033 stage (b) and the oracle mutants of D-035. Decide with the owner after green whether to run it or to close the milestone without it; until then no mutation number is claimed.

## Open questions

- None blocking.
- For the owner's decision: do you want a draft pull request now, to get a Linux CI run of the oracle and timeout tests? It would be red until your six functions exist, and it departs from D-033.3 (pull request when green). Nothing was opened (D-037).
- For the owner's information: `docs/BRIEF.md` §4 still says "Python 3.11+". D-027 records the deviation; edit the brief if you want it to match.
