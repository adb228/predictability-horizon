# CLAUDE.md

Predictability Horizon: a double-pendulum ground truth, then learned models measured in Lyapunov times. Stage 1 is in progress (milestones 1 to 4 of 7 done).

## Read first
`docs/BRIEF.md` (the owner's brief, the authority), `PROGRESS.md` (the handoff section first), `docs/plans/stage1.md` (milestones and gates), `docs/DECISIONS.md` (the log; where it differs from PROGRESS.md, the log wins).

## Standing rules
- **Roles.** The owner derives and verifies the math; the agent implements. Explain the math first, briefly; for the Lyapunov estimators the owner derives before any code (D-039). Say "I'm not certain" where true.
- **No fabricated results.** Every number comes from a committed script; numbers from throwaway labs are labelled "uncommitted lab" and their scripts are not committed.
- **Larger decisions** (costly to reverse, can change numerical results, a new dependency, a public-surface change, any tolerance or gate change) are logged in `docs/DECISIONS.md` before implementing and reviewed by the `reviewer` subagent (design before code, diff after CI is green). Merge only on its explicit `APPROVED FOR MERGE`.
- **Predeclared values and gates** (`src/horizon/tolerances.py`, the register tests): if a gate fails, stop and report. Never adjust a threshold, grid, horizon or seed stream to make it pass.
- **The log is append-only.** Pushed entries are never edited; a correction is a new entry. Next free number: D-073.
- **History.** No force-push, rebase or amend of anything pushed. Identity: the noreply address in `git config user.email`. Never write the owner's university e-mail, tokens or local absolute paths into any file, commit message or PR text. The commit trailer is Claude Code's default.
- **Merges.** Merge commit, never squash, made locally with `git merge --no-ff` and the noreply identity (D-064); do not use `gh pr merge`. The repository's visibility and tags are the owner's.
- **Tests.** New tests are `fast` by default and must be cheap, or be named in `tests/tiers.py` (D-066 to D-071). The fast tier is a smoke tier, never a gate. Per-test timeout 60 s.

## Commands
```
uv sync --locked
uv run --locked pytest -m fast      # about 30 s
uv run --locked pytest -n auto      # full tier, 613 tests
uv run --locked ruff check . && uv run --locked ruff format --check .
```

## Permissions (D-072)
- **Without asking:** tests, Ruff, branch commits and pushes, draft PRs, docs-only pushes to `main`, and merging when CI is green and the reviewer has approved, then deleting the branch.
- **Ask first:** CI or workflow changes, gate or tolerance changes, anything account-level.
- **Never:** force-push.

Milestone 5 (Lyapunov estimators) starts with the math walkthrough; see the end of the handoff in `PROGRESS.md`.
