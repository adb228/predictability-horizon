# CLAUDE.md

Predictability Horizon: a double-pendulum ground truth, then learned models measured in Lyapunov times. Stage 1 is in progress (milestones 1 to 4 of 7 done).

## Read first
`docs/BRIEF.md` (the owner's brief, the authority), `PROGRESS.md` (the handoff section first), `docs/plans/stage1.md` (milestones and gates), `docs/DECISIONS.md` (the log; where it differs from PROGRESS.md, the log wins).

## Standing rules
- **Roles.** The owner derives and verifies the math; the agent implements. Explain the math first, briefly; for the Lyapunov estimators the owner derives before any code (D-039). Say "I'm not certain" where true.
- **No fabricated results.** Every number comes from a committed script; numbers from throwaway labs are labelled "uncommitted lab" and their scripts are not committed.
- **Larger decisions** (costly to reverse, can change numerical results, a new dependency, a public-surface change, any tolerance or gate change) are logged in `docs/DECISIONS.md` before implementing and reviewed by the `reviewer` subagent (design before code, diff after CI is green); see Reviews below. Merge only on its explicit `APPROVED FOR MERGE`. A decision that touches the owner's accounts or personal details is not sent to the reviewer; it goes to the owner.
- **Predeclared values and gates** (`src/horizon/tolerances.py`, the register tests): if a gate fails, stop and report. Never adjust a threshold, grid, horizon or seed stream to make it pass.
- **The log is append-only.** Pushed entries are never edited; a correction is a new entry. Next free number: D-088.
- **History.** No force-push, rebase or amend of anything pushed. Identity: the noreply address in `git config user.email`. Never write the owner's university e-mail, tokens or local absolute paths into any file, commit message or PR text. The history scan of D-065 (`scripts/history_scan.sh`, D-075) allows only four e-mail addresses: the noreply identity, `cursoragent@cursor.com` (old trailer), `noreply@anthropic.com` (the current trailer) and `noreply@github.com` (GitHub's public committer address, added by the owner in D-075). The commit trailer is Claude Code's default.
- **Merges (D-064).** Merge commit, never squash, made locally: `git merge --no-ff <branch> -m "Merge pull request #N from adb228/<branch>" -m "<title>"`. Before `git push origin main`, check that author and committer e-mail are the noreply address, that there are two parents, and that the tree equals the branch head. Do not use `gh pr merge` or the web button. The repository's visibility and tags are the owner's.
- **Tests.** New tests are `fast` by default and must be cheap, or be named in `tests/tiers.py` (D-066 to D-071). The fast tier is a smoke tier, never a gate. Per-test timeout 60 s.

## Reviews
- Start a **new** `reviewer` subagent for every review; never continue an earlier one. It has `Read`, `Grep` and `Glob` only.
- The main session writes the review input to a file in its scratchpad directory (never in the repository): the head SHA, `git diff main...<head>`, the commit identities (`git log --format` with author and committer e-mail), the bodies of the commits (`git log --format=%B main..<head>`), the pull request's title, body and comments (`gh pr view <N> --json title,body,comments`), the diff of `docs/DECISIONS.md` since the earlier pushed head of the branch (it must add entries and remove none), and for every CI run its id, event, job conclusions, test counts, timings and xdist worker count. Pass the file's path in the prompt; the working tree must be at the head.
- A design review gets the logged entries instead of a diff. `APPROVED FOR MERGE` is only valid for the head SHA that was reviewed.

## Commands
```
uv sync --locked
uv run --locked pytest -m fast      # about 30 s
uv run --locked pytest -n auto      # full tier, 613 tests
uv run --locked ruff check . && uv run --locked ruff format --check .
```

## Permissions (D-072, D-073, D-075)
- **Without asking:** tests, Ruff, branch commits and pushes (`git push origin <branch>`), draft PRs, docs-only pushes to `main`, and merging when CI is green and the reviewer has approved, then deleting the branch. The settings cannot test those conditions, so the common spellings of a push to `main`, `git merge` and tags prompt; other spellings do not, and the instruction is the guard.
- **Ask first:** CI or workflow changes, gate or tolerance changes (edits to the frozen gate files are meant to prompt; verified for `tolerances.py` only, D-086), anything account-level.
- **Never:** force-push. There is no GitHub ruleset (D-075): the guards are the instructions in this file, and the deny list in `.claude/settings.json` is a backstop that does not catch every spelling.

Milestone 5 (Lyapunov estimators) starts with the math walkthrough; see the end of the handoff in `PROGRESS.md`.
