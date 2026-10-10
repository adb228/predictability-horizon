# Progress

Running log: what is done, what is next, open questions. Updated at the end of every session (brief §8).

## Where we are (2026-10-10)

Stage 1, milestone 1 of 7 (repository baseline): **locally passed, CI pending.** No scientific results exist yet.

## Done

- Calibration, and the Stage 1 plan approved with amendments (D-001 to D-012).
- Milestone 1 decisions reviewed by Opus 5.5 before implementation (D-013 to D-020, D-024). Owner confirmations recorded (D-021 to D-023).
- Repository baseline: brief and decision log, project scaffold, tests, CI workflow, this log.
- Local gate, run on both locked stacks with the same commands CI uses:

  | Python | jax | numpy | scipy | `ruff check` | `ruff format --check` | `pytest` |
  |---|---|---|---|---|---|---|
  | 3.12 (reference) | 0.11.2 | 2.5.3 | 1.18.1 | pass | pass | 5 passed |
  | 3.11 | 0.10.2 | 2.4.6 | 1.17.1 | pass | pass | 5 passed |

- Two mutation checks showed that the float64 test and the fatal-warning test can fail (D-015).

## Next

- Milestone 1: create the private GitHub remote and push (D-023), confirm CI passes on both Python versions, then mark the milestone done.
- Milestone 2: derive and cross-check the double-pendulum mechanics. The owner writes the equations of motion (D-003); the implementer first walks through the math, supplies the interface and the SymPy-based tests, then reviews.

## Open questions

- None blocking. For the owner's information (D-024): raising `requires-python` to `>=3.12` would collapse the lockfile to a single numerical stack.
