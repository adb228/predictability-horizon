# Decision log

Short log of choices, alternatives, and why (brief §6.7: no silent tuning; every non-obvious choice is recorded here). Entries are append-only. A changed decision gets a new entry that supersedes the old one.

## Approval tags

| Tag | Meaning |
|---|---|
| **Owner** | Decided by the repo owner. |
| **Opus 5.5** | A larger decision the implementer (Sonnet 5.5) deferred to Opus 5.5 under the owner's standing instruction (D-007). The entry records the question, the recommendation sent, and the verdict as returned, and is written **before** the decision is implemented. |
| **Implementer** | Minor decision made by the implementer without deferral; logged for traceability. |

**What counts as "larger"** (items a to d are the implementer's reading of D-007; the owner may correct them): a choice that (a) is costly to reverse once later stages build on it, (b) can silently change numerical results, (c) adds a dependency or tool, or (d) changes the public surface of the repo. Added by the owner in D-025: (e) any change to a predeclared tolerance or gate threshold.
**Not deferred to Opus 5.5:** anything touching the owner's external accounts (creating or publishing a remote, pushing, tagging) or personal details. A model's approval cannot stand in for the owner's authorization, so these go to the owner.

---

## Calibration decisions (Owner, 2026-10-10)

### D-001 Framework: JAX
- **Choice:** JAX (Equinox and Optax from Stage 2). SciPy DOP853 is the independent reference integrator. SymPy is used only to cross-check derivations.
- **Alternative:** PyTorch + `torchdiffeq`.
- **Why:** `jax.jacfwd` gives exact Jacobians for the variational Lyapunov method, and `vmap` batches many initial conditions, all in one stack.

### D-002 Pace
- **Choice:** intensive, 15+ h/week. Targets: Stage 1 in about 3–4 weeks, Stage 2 in about 6–8 weeks. I read the Stage 2 target as counted from project start; owner to correct if not.
- **Alternatives:** 8–10 h/week, 4–6 h/week, no dates.
- **Why:** owner's availability. Advancement is gated by passing evidence, not by date.

### D-003 Who writes what
- **Choice:** the owner writes the equations of motion and energy (`systems/double_pendulum.py`), the Lyapunov estimators (`lyapunov.py`), and the Stage 2 metrics (`metrics.py`: VPT and matched-perturbation baseline). The implementer explains the math first, supplies interfaces and tests, and reviews. The implementer writes everything else (integrators, data generation, plotting, tests, CI).
- **Why:** these are the pieces the owner must be able to defend unaided (brief §0, §7).

### D-004 Depth of math walkthroughs
- **Choice:** conceptual explanation with the final equations and the tests that guard them. The written derivations in `docs/derivations/` still contain the full derivation that brief §3 requires.
- **Alternatives:** key steps and invariants; full line-by-line.
- **Why:** owner preference; keeps chat explanations short (brief §0).

### D-005 Names, license, CI
- **Choice:** GitHub repo `predictability-horizon`, Python package `horizon`, MIT license, GitHub Actions running tests and lint from day one. The local working directory stays `worldmodel`.
- **Alternatives:** keep `worldmodel` as the repo or package name; Apache-2.0; add CI later.
- **Why:** matches brief §5; MIT is permissive and standard for research code.

### D-006 Symplectic integrator: deferred
- **Choice:** not built in Stage 1.
- **Why:** the Hamiltonian is non-separable (the kinetic term depends on `θ1−θ2` through the mass matrix), so explicit Störmer–Verlet does not apply. Implicit midpoint would, but it needs a nonlinear solve per step and is not needed by the reference (DOP853) or fast (RK4) paths. Revisit after core validation if it earns its place.

---

## Stage 1 plan: approval and amendments (Owner, 2026-10-10)

The seven-milestone Stage 1 plan was approved with the amendments D-008 to D-012, and with the standing instruction D-007.

### D-007 Approval delegation
- **Choice:** from the start of implementation, the implementer (Sonnet 5.5) defers larger decisions and questions to Opus 5.5 for approval and logs each deferral here before implementing it.
- **Why:** owner instruction. See "Approval tags" above for how entries are recorded and what is escalated to the owner instead.

### D-008 Variational/QR spectrum in canonical coordinates
- **Choice:** run the pendulum's tangent-space/QR spectrum in canonical `(q, p)`. Any `(θ, ω)` run is a diagnostic only, and must show the exponent sum converging to zero as integration time grows.
- **Why:** the Hamiltonian flow preserves phase-space volume in `(q, p)`, so exponents pair and sum to zero. In `(θ, ω)` the invariant measure is `det M(θ) dθ dω` (M is the mass matrix), so the finite-time exponent sum is `(1/T) ln[det M(θ₀) / det M(θ_T)]`: nonzero, decaying like `1/T`. This is my derivation, not yet checked; milestone 5 must verify it numerically.

### D-009 Estimator validation systems
- **Choice:** before touching the pendulum, validate the Lyapunov estimators on (i) a linear system, whose exponents equal the real parts of the eigenvalues of its matrix; (ii) a harmonic oscillator, with spectrum `(0, 0)`; (iii) Lorenz, with `λ₁ ≈ 0.906` and exact exponent sum `−(σ + 1 + β)`.
- **Assumption to confirm:** Lorenz with the standard parameters `σ = 10, ρ = 28, β = 8/3`, which is where `λ₁ ≈ 0.906` comes from.
- **Why:** known answers catch estimator bugs independently of the pendulum.

### D-010 Wrap differences, never state
- **Choice:** integrated angles are never wrapped. Wrapping, `wrap(Δθ) = ((Δθ + π) mod 2π) − π`, is applied only when computing differences.
- **Why:** wrapping the state introduces discontinuities that break adaptive step control and tangent-space propagation. Metrics only ever need differences.

### D-011 Milestone 6 sensitivity design
- **Choice:** one-at-a-time sweeps around a declared default (timestep, perturbation size, renormalization/QR cadence, transient length, window length, norm), plus one combined worst-case run. No full grid.
- **Alternative:** full factorial grid.
- **Why:** cost. The grid grows multiplicatively in the number of factors. The worst-case run is the check for interactions that one-at-a-time sweeps miss.

### D-012 Repository visibility
- **Choice:** the repository becomes public after the milestone 4 gate passes.
- **Note:** this is an external action on the owner's account. No remote exists yet; see `PROGRESS.md` open questions.

---

## Milestone 1: decisions deferred to Opus 5.5 (2026-10-10)

Deferred under D-007 and logged **before** implementation. Reviewer: Opus 5.5 (subagent `a6ca5197-6ba5-4ead-9e23-87e5e2d7d9eb`). Its first reply reached the implementer as a summary only (one approval, seven amendments, no rejections). The verdicts below are from its follow-up reply, which it stated was unchanged. Each entry shows the resulting choice and what Opus 5.5 changed from the implementer's recommendation.

### D-013 Packaging and layout (M1-1, AMEND)
- **Choice:** project `predictability-horizon`; import package `horizon` in `src/horizon/`; `[build-system] requires = ["uv_build>=0.8.22,<0.9"]`, `build-backend = "uv_build"`, `[tool.uv.build-backend] module-name = "horizon"`; `[tool.uv] required-version = ">=0.8.22"`; `requires-python = ">=3.11"`; `.python-version` = 3.12; `uv.lock` committed; no placeholder modules or directories.
- **Opus 5.5 changed:** added the `uv_build` upper bound and the minimum uv version.
- **Alternatives:** `hatchling`; flat layout; no lockfile.
- **Why:** an unbounded backend lets a future release change the build with no change in the repo; an older uv could write a different lockfile; src layout stops accidental imports from the working directory.
- **Superseded in part by D-027:** `requires-python` is now `>=3.12`.

### D-014 Dependencies are added when first imported (M1-2, AMEND)
- **Choice:** milestone 1 adds runtime `jax`, `numpy` and dev `ruff` (commit 2), then dev `pytest`, `hypothesis` (commit 3). Later, each in the commit that first imports it: `sympy` (dev) at milestone 2, `scipy` (runtime) at milestone 3, `matplotlib` (runtime) at milestone 7. Declaring `numpy` explicitly is approved. Not added: `equinox`, `optax`, `diffrax`, GPU JAX, PyYAML, mypy/pyright, pre-commit, pytest-cov, pytest-xdist.
- **Opus 5.5 changed:** staggered the dependencies instead of adding them all at once; ruled that declaring `numpy` needs no owner approval.
- **Why:** keeps the history honest and the environment minimal. `numpy` is a direct import of a package already in the stack's transitive dependencies.

### D-015 float64 policy (M1-3, AMEND)
- **Choice:** `src/horizon/__init__.py` calls `jax.config.update("jax_enable_x64", True)` on import. Smoke test: a fresh interpreter, `subprocess.run([sys.executable, "-c", ...], check=True)`, imports only `horizon` and `jax.numpy` and asserts `jnp.zeros(()).dtype == jnp.float64`, `jnp.asarray(1.0).dtype == jnp.float64`, `jnp.finfo(jnp.asarray(1.0).dtype).eps < 1e-15`, and `bool(jnp.asarray(1.0) + 1e-12 != jnp.asarray(1.0))`. Never test with plain Python floats. `tests/conftest.py` and every script and entry point import `horizon` before anything else from JAX. Before relying on the flag, check the locked JAX version's changelog and docs that `jax_enable_x64` is not deprecated, and record the result below.
- **Opus 5.5 changed:** the test design, the import-order rule, and the deprecation check. My proposed check (`1.0 + 1e-12 != 1.0` on Python floats) can never fail; an in-process test can pass because `conftest.py` already enabled float64; arrays created before `horizon` is imported stay float32.
- **Alternatives:** env var `JAX_ENABLE_X64=1` (easy to forget, then silently float32); explicit `horizon.enable_x64()` at every entry point.
- **Known cost:** an import-time global side effect. Revisit at Stage 3, when GPU networks will want float32.
- **Implementer note:** ruff's isort rule sorts third-party `jax` imports before first-party `horizon`, so I read the import-order rule as "before any JAX array is created or traced" (importing `jax` itself is harmless). It holds strictly in `conftest.py`, which imports no JAX, and conftest loads before every test module.
- **Deprecation check result (2026-10-10, both locked versions per D-024):** passed. (a) `python -W error -c "import horizon"` raised no warning and gave a float64 default on jax 0.10.2 (Python 3.11) and jax 0.11.2 (Python 3.12). (b) The changelogs at tags `jax-v0.10.2` and `jax-v0.11.2` deprecate only the experimental context managers `jax.experimental.enable_x64` and `disable_x64`, in favor of `jax.enable_x64`. The config flag `jax_enable_x64` that we set is not listed as deprecated, and in the jax 0.11.2 source it is an ordinary config state. We use none of the experimental context managers; if Stage 3 needs a scoped toggle, use `jax.enable_x64`. (c) The docs page was not part of this check: fetching `docs.jax.dev/en/latest/default_dtypes.html` returned no match.
- **Implementer additions within the approved test design:** the fresh-interpreter helper removes `JAX_ENABLE_X64` from the child environment, so a developer's shell cannot make the test pass vacuously. A negative-control test shows that without `import horizon` JAX truncates a float64 request and emits the warning that D-018 makes fatal; it also fails if a JAX upgrade rewords that warning.
- **Mutation checks (2026-10-10), to show the tests can fail:** (A) with the `jax_enable_x64` line removed, only the float64 test fails, at the dtype assertion. (B) with the fatal filter overridden by `-W ignore`, only the filter test fails (`DID NOT RAISE UserWarning`). Both mutations were reverted.

### D-016 Continuous integration (M1-4, AMEND)
- **Choice:** `.github/workflows/ci.yml`; triggers: push to `main` and all pull requests; `ubuntu-latest`; Python matrix 3.11 and 3.12; `permissions: contents: read`; `timeout-minutes: 20`; `concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }`. Steps: `actions/checkout`, then `astral-sh/setup-uv` with `version: "0.8.22"` and `enable-cache: true`, `uv sync --locked --python <matrix version>`, `uv run ruff check .`, `uv run ruff format --check .`, full `uv run pytest`. No macOS runner. Use the current latest major versions of both actions, looked up before committing, not the v4/v5 I proposed from memory.
- **Opus 5.5 changed:** action versions must be looked up rather than recalled; pinned uv version; job timeout; concurrency group.
- **Alternatives:** single Python version; macOS arm64 runner; pin actions by commit SHA.
- **Why:** with no remote, "verify on the first real run" cannot happen soon. One OS with a Python matrix tests the `>=3.11` claim at low cost.
- **Action versions used (looked up 2026-10-10 from the GitHub releases API):** `actions/checkout@v7` (latest release v7.0.1; the floating tag `v7` exists) and `astral-sh/setup-uv@v10.3.0` (latest release; **no floating `v10` tag exists**, so the exact release tag is pinned, which is the same major Opus 5.5 asked for). The versions I proposed from memory, v4 and v5, were two and five majors behind.
- **Implementer adaptations that keep the approved intent:** (1) the Python version is set by a job-level `UV_PYTHON` instead of `--python` on `uv sync` alone. Verified locally: after `uv sync --python 3.11`, a plain `uv run` deleted the 3.11 environment and rebuilt it as 3.12 because of `.python-version`, so the literal step list would have run the "3.11" job on Python 3.12. (2) The job prints the jax, jaxlib, numpy and scipy versions and asserts the interpreter matches the matrix entry, so a mismatch fails loudly. (3) Every `uv run` uses `--locked`, stricter than the literal list.
- **Verified so far:** the workflow parses as YAML, and both matrix legs were simulated locally with the same commands and `UV_PYTHON` (ruff clean, 5 of 5 tests pass on jax 0.10.2 and on jax 0.11.2). **Not yet verified:** an actual GitHub Actions run.
- **Superseded in part:** the runner image by D-026; the matrix, `UV_PYTHON` and the interpreter assertion by D-027.

### D-017 Lint and format (M1-5, APPROVE)
- **Choice:** ruff, `line-length = 100`, `target-version = "py311"`, select `E, F, W, I, UP, B, SIM, NPY, D, ANN`, numpy docstring convention; `D` and `ANN` relaxed only under `tests/`.
- **Why:** `D` + `ANN` mechanically enforce brief rule 8 (type hints and docstrings on public functions); neither checks units or array shapes, which stay a review item. `NPY` flags legacy global-seed NumPy randomness, which supports the explicit-seed policy.
- **Superseded in part by D-027:** `target-version` is now `py312`.

### D-018 Test infrastructure and determinism (M1-6, AMEND)
- **Choice:** pytest config: `testpaths = ["tests"]`, `addopts = "-ra --strict-markers"`, `xfail_strict = true`, registered marker `slow` (CI still runs everything), `filterwarnings = ["error:Explicitly requested dtype:UserWarning"]`. `tests/conftest.py` registers Hypothesis profile `default` (`derandomize=True, deadline=None, database=None, print_blob=True`) and profile `explore` (`derandomize=False, deadline=None, max_examples=1000`), and loads `settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))`.
- **Opus 5.5 changed:** added `xfail_strict`, the fatal JAX-dtype warning, `database=None`, `print_blob=True`, and the environment-variable profile selector.
- **Why:** that JAX warning is what silent float32 truncation looks like at runtime. Derandomized runs are reproducible only for a fixed Hypothesis version, which `uv.lock` provides. Without a database, a run cannot replay examples saved by earlier local runs.
- **Implementer additions:** two tests guard the configuration itself. One asserts the registered `default` Hypothesis profile is deterministic (`derandomize`, `deadline=None`, `database=None`), so a silent edit fails a test. The other raises a synthetic dtype-truncation warning to prove the fatal `filterwarnings` entry is active.

### D-019 Randomness, config and provenance convention (M1-7, AMEND)
- **Choice (documentation only in milestone 1; no `config.py` or seeding helper yet):** every stochastic script takes an explicit integer seed from a committed frozen-dataclass config, written to JSON next to its results. JAX randomness uses `jax.random.key(seed)` and `split`; NumPy uses `np.random.default_rng(seed)`; never global seeding. Every result JSON also records the git commit SHA, a dirty-working-tree flag, the versions of `python`, `jax`, `jaxlib`, `numpy`, `scipy` (once added) and `hypothesis` where relevant, and `platform.platform()`.
- **Opus 5.5 changed:** added the provenance fields.
- **Alternatives:** build `horizon/config.py` now; YAML configs.
- **Why:** a seed alone cannot reproduce a number when XLA or SciPy versions differ. A config module's shape should come from the first real experiment.

### D-020 Commit plan (M1-8, AMEND)
- **Choice:** local commits only; no push, tag, or remote creation without the owner. (1) brief + decision log, including these verdicts, so each decision is logged before the change it governs; (2) scaffold: `pyproject.toml`, `uv.lock` (only `jax`, `numpy`, `ruff`), package, license, readme, gitignore, `.python-version`, ruff passing; (3) test infrastructure and smoke tests: adds `pytest` and `hypothesis` to the dev group and updates `uv.lock` in the same commit, pytest passing (pytest exits with code 5 when it collects no tests); (4) CI workflow; (5) progress log.
- **Opus 5.5 changed:** the order, and what each commit must pass, so every commit is green on its own.

### Risks raised by Opus 5.5 and how each is handled
- **Git identity becomes public** (D-012): no commit until the owner confirms the author name and email (confirmed: D-021).
- **Milestone 1 cannot fully pass yet:** "passes in CI" needs a remote. `PROGRESS.md` records "locally passed, CI pending", not done.
- **Local runs may not match the lockfile:** plain `uv run` can silently re-resolve. The README reproduce command is `uv sync --locked` followed by `uv run --locked pytest`.
- **Future suite runtime:** the `t = 1000` DOP853 energy-drift test will run once per Python version in CI. Measure it at milestone 4 and raise `timeout-minutes` rather than skipping it as `slow`.

### Escalated to the owner (resolved 2026-10-10)
1. Create the GitHub remote, because the CI half of the milestone 1 gate is blocked until it exists: see D-023.
2. Confirm the MIT LICENSE holder name and the git author name and email before the first commit: see D-021 and D-022.

---

## Owner confirmations escalated by Opus 5.5 (Owner, 2026-10-10)

### D-021 Git author identity
- **Choice:** name `adb228`, email `172058618+adb228@users.noreply.github.com` (the GitHub noreply address), set in this repo's local git config only. The global config is unchanged.
- **Alternative:** the global identity `adb228 <adb228@g.ucla.edu>`.
- **Why:** the author name and email become public with the history (D-012). The noreply address keeps the UCLA email out of it.

### D-022 License holder
- **Choice:** `Copyright (c) 2026 adb228`, the same name as the git author identity.

### D-023 GitHub remote
- **Choice:** the implementer creates private `adb228/predictability-horizon` with `gh` and pushes `main` so CI runs. It stays private until the milestone 4 gate. Making it public is a separate step (D-012).
- **Alternatives:** the owner creates it; hold, with the gate recorded as "locally passed, CI pending".
- **Why:** the CI half of the milestone 1 gate cannot pass without a remote.
- **Outcome (2026-10-10):** the private repo was created. The first push was rejected because the `gh` OAuth token lacked the `workflow` scope, which GitHub requires to create or update `.github/workflows/*`. The owner chose, from the options (grant the scope, switch to SSH, or hold), to grant it. I ran `gh auth refresh -h github.com -s workflow` and the owner approved the one-time code in the browser. **The token's scopes are now `gist`, `read:org`, `repo`, `workflow`** (previously without `workflow`). `main` was pushed and CI run 38039152732 passed on Python 3.12 (19 s) and 3.11 (21 s).

---

## Follow-up review by Opus 5.5 (2026-10-10)

Consulted through a follow-up to the same reviewer, after `uv add` revealed a fact the first review did not have.

### D-024 Two locked stacks; Python 3.12 is the reference (Opus 5.5, AMEND)
- **Superseded by D-027 (2026-10-10):** the owner raised `requires-python` to `>=3.12`, so there is one locked stack. The entry is kept as the record of why.
- **Fact (verified from PyPI metadata and `uv.lock`):** jax 0.11.2 requires Python `>=3.12` and jax 0.10.2 requires `>=3.11`, so the lock forks. Python 3.11: jax and jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1. Python `>=3.12`: jax and jaxlib 0.11.2, numpy 2.5.3, scipy 1.18.1. The CI matrix (D-016) therefore tests the oldest and the newest stack, not one stack twice. `uv add` wrote the lower bounds `jax>=0.10.2` and `numpy>=2.4.6`, the minimum across the forks.
- **Choice:** keep `requires-python = ">=3.11"` (brief §4) and the 3.11 + 3.12 matrix. Python 3.12 (`.python-version`, jax 0.11.2) is the **reference environment**: every committed result and figure is generated there, and each result JSON records library versions (D-019). The 3.11 job is a lowest-supported-version check. All tests must pass on both stacks with tolerances declared up front. A test that fails on only one stack blocks the milestone and must be investigated; it is never skipped or loosened per stack. The README reproduce commands pin `--python 3.12`.
- **Opus 5.5 changed:** pinned the README commands to the reference stack and added the one-stack-failure rule. It also ruled that the D-015 deprecation check must cover both locked JAX versions.
- **Alternatives:** raise `requires-python` to `>=3.12` (a single stack, but it deviates from the brief, so it is the owner's call); cap `jax<0.11` (forgoes current JAX).
- **Why:** without the pin, a user whose default Python is 3.11 would silently regenerate results on the older stack.
- **For the owner's information:** raising `requires-python` to `>=3.12` would collapse the lockfile to one numerical stack.

---

## Owner decisions after milestone 1 (Owner, 2026-10-10)

### D-025 Deferral rule extended: tolerance and gate changes go to Opus 5.5
- **Choice:** any change to a predeclared tolerance or gate threshold goes to Opus 5.5 for approval before it is made, and is logged here first (D-007). A tolerance or threshold is *predeclared* once it is fixed by the brief, or recorded in this log before the test or script it governs is first run against the code under test. The rule covers loosening and tightening, thresholds taken from the brief (for example the energy-drift gate in §3), and tolerances declared in later milestones.
- **Why:** a gate that can be adjusted after a failure is not a gate (brief §6.2, §6.7).
- **Effect:** adds item (e) to "What counts as larger" at the top of this file.

### D-026 CI runner pinned to `ubuntu-24.04`
- **Choice:** the CI job runs on `ubuntu-24.04`. Supersedes `ubuntu-latest` in D-016.
- **Why:** GitHub moves `ubuntu-latest` to Ubuntu 26 starting 2026-10-19, so the environment would change with no change in the repo. A newer image is adopted deliberately and logged.

### D-027 Python 3.12 or newer only; CI collapses to one job
- **Choice:** `requires-python = ">=3.12"`, and CI runs a single job on Python 3.12. Supersedes `>=3.11` (D-013), the 3.11 + 3.12 matrix (D-016), and D-024.
- **Deviates from brief §4 ("Python 3.11+").** `docs/BRIEF.md` was not edited, because it is the owner's document; this entry is the record of the change.
- **Why:** jax 0.11 requires Python 3.12 or newer (D-024), so supporting 3.11 forked the lockfile into two numerical stacks. One stack means one set of numbers to defend.
- **Consequences, all mechanical:** `uv.lock` collapses to the single stack that was the reference environment; the dependency lower bounds are raised to the locked, tested versions (`jax>=0.11.2`, `numpy>=2.5.3`), since the 3.11 job that tested the old bounds is gone; ruff `target-version` becomes `py312`; the CI matrix, the `UV_PYTHON` workaround and the interpreter assertion are removed, because they only guarded the multi-version run (the job still prints library versions); the README commands drop `--python 3.12`; the one-stack-failure rule of D-024 no longer applies.
- **Verified after the change (2026-10-10):** `uv lock` removed only the four 3.11-fork entries (jax and jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1) and changed no remaining version: jax and jaxlib 0.11.2, numpy 2.5.3, scipy 1.18.1, hypothesis 6.168.5, pytest 9.1.1, ruff 0.17.0. From a clean `.venv`, `ruff check`, `ruff format --check` and `pytest` (5 passed) pass locally, and the workflow YAML parses with one job, `runs-on: ubuntu-24.04`, and no matrix. A GitHub run is pending the push.
