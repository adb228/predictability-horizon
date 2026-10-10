# Progress

Running log: what is done, what is next, open questions. Updated at the end of every session (brief §8).

## Where we are (2026-10-10)

Stage 1, milestone 1 of 7 (repository baseline): **done.** The gate passed locally and in CI on the private remote. Since then the project requires Python 3.12 or newer, with one locked stack and one pinned CI job (D-026, D-027). No scientific results exist yet.

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

## Next

- Milestone 2: derive and cross-check the double-pendulum mechanics. The owner writes the equations of motion (D-003); the implementer first walks through the math, supplies the interface and the SymPy-based tests, then reviews.

## Open questions

- None blocking.
- For the owner's information: `docs/BRIEF.md` §4 still says "Python 3.11+". D-027 records the deviation; edit the brief if you want it to match.
