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
- **Update (D-039, broadened the same day):** the agent implements all code and the owner derives and verifies, so the owner no longer writes the three pieces named above. The agent still walks the owner through the math before writing the code (brief §7), and for the Lyapunov estimators the owner derives and verifies it first. Brief §0 is unchanged.

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
- **Verified after the change (2026-10-10):** `uv lock` removed only the four 3.11-fork entries (jax and jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1) and changed no remaining version: jax and jaxlib 0.11.2, numpy 2.5.3, scipy 1.18.1, hypothesis 6.168.5, pytest 9.1.1, ruff 0.17.0. From a clean `.venv`, `ruff check`, `ruff format --check` and `pytest` (5 passed) pass locally, and the workflow YAML parses with one job, `runs-on: ubuntu-24.04`, and no matrix. GitHub Actions run 38040776944 passed in one job on image `ubuntu-24.04` with CPython 3.12.3 and the same library versions (17 s), and no `ubuntu-latest` migration annotation.

---

## Milestone 2: decisions deferred to Opus 5.5 (2026-10-10)

Deferred under D-007 and logged **before** any milestone 2 code, test or test case exists. Reviewer: Opus 5.5 (subagent `a6ca5197-6ba5-4ead-9e23-87e5e2d7d9eb`), in two exchanges: the design review (proposals P1 to P6 and open questions Q7), and a follow-up about one check that proved infeasible (D-032). Both replies returned the full verdict block. Each entry shows the resulting choice and what Opus 5.5 changed from the implementer's recommendation. D-025 applies from this commit: every value below is predeclared, and any later change goes to Opus 5.5 first.

**Evidence behind the review: uncommitted pre-freeze feasibility checks, not results.** Throwaway scripts in `/tmp` (a SymPy prototype oracle and an unrelated reference solution of the owner-authored module, run with `uv run --no-project`; arbitrary seeds 0, 1, 12345, 777, 4242 and 99, none of them the frozen seed). They are not committed because they only informed design choices; every milestone 2 result will come from committed tests. What they showed:
- The brief's `T` and `V` equal the Cartesian-geometry ones symbolically, and the oracle's mass matrix and forcing terms equal my hand derivation symbolically. At 60 digits the oracle's own Euler-Lagrange residual is about 1e-59 (500 states, 20 parameter sets).
- The float64-lambdified SymPy expression is 1.1e-12 off the 60-digit value (relative, scale floor 1e-3), worse than a plain JAX solve (8.9e-14). It must not be used as ground truth, so the oracle evaluates at 60 digits.
- Against the frozen tolerance classes, the reference solution's observed error divided by the allowed error (below 1 passes) was: mass matrix 1e-5, energies 2e-5, acceleration 5e-4, Euler-Lagrange residual 9e-5, Jacobian 1e-4, jit+vmap versus eager 9e-4, moderate-angle periodicity 0.028, scaling invariances at most 4e-3, equilibria 4e-4, normal modes 5e-5, energy rate 3e-6. Symmetries were exact. The periodicity check at |θ| up to 1e3 was the exception: see D-032.

### D-028 Public interface of the double-pendulum module (P1, APPROVE)
- **Choice:** module `horizon.systems.double_pendulum` (with `horizon/systems/__init__.py`). State: array of shape `(4,)`, order `(θ1, θ2, ω1, ω2)`, radians from the downward vertical (`θ = 0` hangs down), integrated angles never wrapped (D-010). Functions act on one state; batching is done by the caller with `jax.vmap`. `Params` is a `typing.NamedTuple` `(m1, m2, L1, L2, g)` with defaults `1.0` (nondimensional units, time unit `√(L/g)`), already a JAX pytree; `DEFAULT_PARAMS = Params()`. `params` is a **required** argument of every function, so a test cannot run on the all-ones defaults, which hide parameter-swap bugs. Owner-written, pure, float64, `jit`/`vmap`/`jacfwd`-compatible: `mass_matrix(theta (2,), params) -> (2, 2)`; `kinetic_energy(state, params) -> scalar`; `potential_energy(state, params) -> scalar`; `energy(state, params) -> scalar` (`T + V`); `acceleration(state, params) -> (2,)` (`α1, α2`); `vector_field(state, params) -> (4,)` = `(ω1, ω2, α1, α2)`. Autonomous: no `t` argument (the milestone 3 SciPy wrapper adapts to `solve_ivp`'s `f(t, y)`).
- **Deferred to milestone 5:** the canonical `(q, p)` map, the Hamiltonian vector field, anything Lyapunov-specific (D-008). Reason: scope, and every documented claim must map to a test.
- **Opus 5.5 changed:** nothing.
- **Alternatives:** an API with leading batch dimensions; a dataclass registered as a pytree.
- **Why:** one state per call leaves no batch-dimension logic to get wrong, and a required `params` makes the accidental all-ones run impossible.

### D-029 Independent oracle and its self-checks (P2, AMEND)
- **Choice:** `tests/oracles/double_pendulum.py`, using only NumPy, SymPy and mpmath (no JAX, so it shares no code with the system under test). It derives from Cartesian geometry: positions, then `T` and `V`, then SymPy Euler-Lagrange, then the solved accelerations. No equations of motion are typed by hand inside the oracle. Authoritative values are evaluated at 60 digits with mpmath and rounded to float64 only at the end. Inputs are converted with `mpmath.mpf(float(x))` and nothing else, and a test asserts that this round-trips exactly. Self-checks, green before the owner writes anything: **O1** the Cartesian `T`, `V` equal the brief's symbolically; **O2** the 60-digit Euler-Lagrange residual of the oracle's accelerations is below `ORACLE_SELF_CHECK_TOL` at the seeded cases; **O3** the derivation document's `M` and `b` equal the oracle's symbolically (this verifies the document itself); **O4** the oracle's linearization at the hanging equilibrium with default parameters has eigenvalues exactly `±i√(2 ± √2)`; **O5** generator guard: every seeded parameter set has all five values pairwise different by at least 10 percent, so a swap among `m1, m2, L1, L2, g` changes the output.
- **Opus 5.5 changed:** (1) `mpmath` is logged as approved on the `numpy` precedent of D-014: a direct import of a package already required by an approved dependency (`sympy`, approved for milestone 2 in D-014). (2) `tests/oracles` must be importable deterministically. (3) The `mpf(float(x))` rule and its round-trip test. (4) O2 is close to circular, since it substitutes the oracle's own solution back in, so it counts only as a check on the solve step. **The independent anchors are O1 (energies equal the brief's), O3 (the document's `M` and `b`) and O4 (normal modes).**
- **Implementation details, confirmed in the follow-up:** import mode is `tests/__init__.py` and `tests/oracles/__init__.py` with imports written `from tests.oracles import double_pendulum`; pytest's default `prepend` mode puts the repo root on `sys.path`, `uv_build` does not install `tests/`, and ruff treats `tests` as first-party with no extra config. The residual's monomials come from `sympy.simplify` followed by `sympy.expand(expr, trig=False)`, stated in the docstring so the monomial set does not depend on SymPy defaults (feasibility check: unsimplified Cartesian residuals have 10 and 7 monomials with sin²+cos² pairs that inflate the scale by up to 16.6×, median 1.2×; simplified ones have 6 and 4, the natural terms of `M α = b`). Following D-014, `sympy` and `mpmath` (dev) are added in the commit that first imports them, the oracle commit, not in a separate commit.
- **Alternatives:** use the float64-lambdified expression as truth (rejected, see the evidence above); include JAX in the oracle.
- **Why:** an oracle that shares code or conventions with the code under test cannot catch their shared mistakes.

### D-030 Owner-code tests (P3, AMEND)
- **Choice:** cases are 20 seeded non-default parameter sets times 25 states, drawn with `np.random.default_rng` and explicit seeds, not Hypothesis (matches the gate's "seeded random states"; the Hypothesis profiles stay for later milestones). Every test names its tolerance class from D-031, and its docstring says what it catches and what it is blind to. Tests, red until the owner implements:
  - **T1** `mass_matrix` versus the oracle.
  - **T2** `kinetic_energy`, `potential_energy` and `energy` versus the oracle's Cartesian `T` and `V`.
  - **T3** `acceleration` versus the oracle's accelerations, on the 500 cases and on five large-angle states (D-032).
  - **T4** the gate's residual check: the Euler-Lagrange residual of the **owner's** acceleration, evaluated by the oracle at 60 digits, within tolerance of the natural scale.
  - **T5** `dE/dt = ∇E · f = 0`, using `jax.grad` of the owner's energy, no integration.
  - **T6** symmetries: acceleration even in `ω` and odd in `θ`; the vector field is `2π`-periodic in each angle (float shift by `2πk`, `k = −1, +1`, moderate cases only).
  - **T7** all four equilibria (`θi ∈ {0, π}`, `ω = 0`): vector field zero; hanging energy equals `−[(m1+m2) g L1 + m2 g L2]`; `E` is at least that at every case.
  - **T8** `jax.jacfwd` of the vector field equals the oracle's SymPy Jacobian at the cases (milestone 5 needs autodiff Jacobians).
  - **T9 (the gate)**, default parameters, at the hanging equilibrium: the Jacobian blocks directly (top-left `[:2,:2]` = 0, top-right `[:2,2:]` = I, bottom-right `[2:,2:]` = 0); the eigenvalues of `−J[2:,:2]` are real and equal `2 − √2` and `2 + √2`; the 4×4 complex eigenvalues are `±i√(2 ± √2)` with real parts about 0 (secondary).
  - **T10** scaling invariances for factors 0.5 and 3.0: scaling both masses leaves acceleration unchanged and scales `E` linearly; `(L, g) → (cL, cg)` leaves acceleration unchanged and scales `E` by `c²`; **gravity time scaling** `acceleration(θ, √k·ω; g → k·g) = k · acceleration(θ, ω; g)`.
  - **T11** contract: shapes, float64 dtype, and `jit` + `vmap` over states **and over a batch of `Params`**, with `params` passed as a traced argument (never closed over or static), equals the per-case result. This catches Python branching on parameter values, which an eager call cannot.
- **Known blind spot, written into the test docstrings:** equilibria and normal modes cannot detect `cos(θ1+θ2)` in place of `cos(θ1−θ2)`, because every equilibrium angle is a multiple of π and the two coincide. Only T1 to T4 and T8 catch it, and they depend on the oracle being right, hence O1, O3 and O4.
- **Opus 5.5 changed:** T9 asserts blocks, not just "real parts about 0"; T10 gains the gravity time-scaling identity (otherwise the document's claim that time is in units of `√(L1/g)` has no test); T11 covers batched traced `Params`; T6 gained five large-angle states (**superseded by D-032**).
- **Implementation detail, confirmed in the follow-up (T9):** `−J[2:,:2]` equals `M⁻¹K`, which is **not symmetric**, so `eigvalsh` would silently read one triangle. The test uses `np.linalg.eigvals`, asserts the largest `|imag|` is within the identity tolerance, sorts the real parts, and compares.
- **Alternatives:** Hypothesis strategies for the cases.

### D-031 Predeclared tolerance register and comparison rule (P4, AMEND)
- **Where:** `src/horizon/tolerances.py`, each constant with a docstring giving its basis. The git history of this file is the D-025 audit trail. Later scripts read it too.
- **One rule, used everywhere:** a check passes when `|x − x_ref| ≤ ATOL + RTOL · s`. For value comparisons (T1, T2, T3, T8, T10, T11, and the T7 and T9 comparisons) `s = max(‖x_ref‖∞, 1.0)` per case. For residual checks `s` is the sum of the absolute values of the individual terms: for T4 and O2, the monomials of the simplified-then-expanded Euler-Lagrange residual component (D-029) evaluated at the case; for T5, the four terms `(∂E/∂s_i) · f_i`. The rule is implemented once, as `check_close(x, x_ref, *, rtol, atol, scale=None)`: it fails on any NaN or inf (written `not (err <= allowed)`, so a NaN cannot slip through), requires exactly equal shapes with no broadcasting, and rejects complex input unless the caller passes it explicitly. It reports the worst offender and its ratio. The module docstring states the rule, and so does the document's claim-to-test table (D-034).
- **Frozen values:**
  - `ORACLE_DIGITS = 60` (a setting, not a tolerance).
  - `ORACLE_SELF_CHECK_TOL = 1e-40`, used as both ATOL and RTOL for O2 (observed about 1e-59 absolute).
  - `IMPL_RTOL = 1e-10`, `IMPL_ATOL = 1e-12`: anything compared with the oracle (T1 to T4, T8), T5 relative to its own scale.
  - `IDENTITY_RTOL = 1e-12`, `IDENTITY_ATOL = 1e-12`: exact identities (T6, T7, T9, T10) and T11.
  - `CASES_SEED = 20261010`, fixed here before any frozen case was generated, never changed afterwards. `N_PARAM_SETS = 20`, `N_STATES = 25`.
  - Parameters: each of `m1, m2, L1, L2, g` log-uniform in `[0.5, 2.0]`, redrawn until every pair satisfies `|a − b| / max(a, b) ≥ 0.10`. States: `θ1, θ2` uniform in `[−4π, 4π]`, `ω1, ω2` uniform in `[−5, 5]`.
  - Draw order for set `k` (stream `np.random.default_rng([CASES_SEED, k])`): the parameter set (`rng.uniform(log 0.5, log 2, size=5)` per attempt, exponentiated), then `θ` with `size=(25, 2)`, then `ω` with `size=(25, 2)`.
  - Five large-angle states (D-032): `θ` uniform in `[−1e3, 1e3]`, `ω` uniform in `[−5, 5]`, parameter sets 0 to 4, drawn from their own stream `np.random.default_rng([CASES_SEED, 1000])` so they do not shift the moderate cases (`θ` with `size=(5, 2)`, then `ω` with `size=(5, 2)`).
  - Reproducibility of the cases is tied to the NumPy version locked in `uv.lock`.
- **Basis, and disclosure:** the values come from a floating-point error analysis and were sanity-checked against an unrelated reference solution (the evidence above), never against the owner's code. An honest float64 implementation measured about 1e-13 relative (margin 1e3 or more at 1e-10), while structural bugs (a sign, a swapped parameter, the wrong coupling) give errors of order 1e-3 to 1 at generic states. Identity checks are limited by a few ulp times `|θ|`.
- **Opus 5.5 changed:** the single comparison rule and its scale definitions (the original proposal left them ambiguous, which is where silent loosening would happen); `CASES_SEED` chosen and logged before case generation; the parameter and state ranges written down.
- **Implementer additions within the approved design:** small tests of `check_close` itself (passes inside the tolerance, fails outside, fails on NaN and inf, fails on shape mismatch, rejects complex input), because a gate helper that cannot fail is not a gate.

### D-032 Large-angle check moved into the oracle comparison (follow-up, Option C)
- **What was superseded:** the T6 change from the design review, "add five states with `|θ|` up to 1e3 to the periodicity check" under the identity tolerances.
- **Why it was infeasible (uncommitted pre-freeze feasibility check against the implementer's reference, not a result):** the check compares `f(θ_big)` with `f(θ_reduced)`, where `θ_reduced` is the exact `2π` reduction computed at 60 digits and rounded once. With the standard `cos(θ1 − θ2)`, correct code fails it for a small fraction of states, because the float64 subtraction `θ1 − θ2` of two numbers near 1e3 already loses up to about 1e-13, and the function amplifies that when `det M` is small. Observed error divided by allowed error, over random states: sample 1 (3000 states, `|θ| ≤ 1e3`): median 0.001, 99th percentile 0.416, max 1.897, 0.17 percent above 1. Sample 2 (2000 states): 99th percentile 0.353, max 0.992, 0.55 percent above 0.5. With `|θ| ≤ 1e2` (2000 states): 99th percentile 0.055, max 0.116. The alternative below, with `|θ| ≤ 1e3`: median 0.0000, 99th percentile 0.0094, max 0.055, none above 0.5. Opus 5.5 noted that the original check was set without this measurement and "was never a fair test".
- **Choice (Option C):** five large-angle states (frozen in D-031) join T3, the oracle comparison under the IMPL class. The exact function is periodic, so agreement with it at large unwrapped angles implies periodicity to the IMPL tolerance, and it also tests correctness there (margin 18 or more). The identity periodicity check (float shift by `2πk`, `k = −1, +1`, each angle) stays on the moderate cases only. **No tolerance constant changes.**
- **Alternatives:** (A) keep the check as approved (about 0.5 percent chance that the five frozen states fail for correct code, with a thin tail margin); (B) cap `|θ|` at 1e2.
- **Known limitation, carried into milestones 3 to 5:** float64 conditioning of unwrapped angles degrades with `|θ|`: about 1e-13 absolute in `θ1 − θ2` at `|θ| = 1e3`, amplified when `det M` is small. It does not change correctness at the tolerances above, but it bears on milestone 3 (RK4 versus DOP853 at long horizons), milestone 4 (energy drift at `t = 1000`) and milestone 5 (tangent-space accuracy).

### D-033 Workflow, test-quality evidence and authorship (P5, AMEND)
- **Choice:**
  1. This log is committed on `main` as a docs-only commit before any code, so the tolerances are timestamped before any test runs against the owner's code.
  2. Branch `milestone-2-mechanics`, with separate commits: derivation document; tolerance register with `check_close` and its tests; oracle with O1 to O5 (green; adds `sympy` and `mpmath`, and the `uv.lock` diff must show only those additions); skeleton (bodies raise `NotImplementedError`, docstrings give units and shapes) with T1 to T11 (red); progress update.
  3. CI triggers only on pushes to `main` and on pull requests, so pushing the branch causes no deliberate red run. The pull request is opened when the owner's implementation is green locally and merged with a **merge commit** (no squash), so the first-parent history of `main` stays green and the red-to-green history is preserved.
  4. **Test-quality evidence, in two stages.** (a) Before the owner starts, I run an uncommitted lab in `/tmp` (a reference solution plus mutants: gravity sign flipped; `cos(θ1+θ2)` in `M` only; `sin(θ1+θ2)` in the Coriolis term only, which preserves equilibria; `m1`/`m2` swapped in one term; `L1`/`L2` swapped in one term; `ω` in place of `ω²` in one term; a sign error in one coupling term; a missing ½ in the kinetic cross term; a sign error in one `V` term; float32 truncation; Python branching on a value; a small damping term; plus two correct variants, closed form versus `linalg.solve` and angle wrapping inside the function, that must still pass). Only the **qualitative outcome** is logged, labelled "uncommitted, pre-implementation". (b) After the owner's code is green, `scripts/mutation_check.py` is committed. It applies each mutant as an exact text substitution to the owner's committed source, asserts each substitution matches exactly once, and asserts the suite fails for every mutant and passes for the unmodified source. The final numbers are logged from that script (brief rule 1: every number comes from a committed script).
  5. The owner never sees the `/tmp` reference solution.
  6. `scripts/` is approved as public surface for **development tools only**. Experiment and figure code still goes in `experiments/` (brief §5).
- **Authorship rules (Opus 5.5, Q7c; no objection to the owner writing these functions; replaced by D-039, the agent implemented them):** the owner writes `mass_matrix`, `kinetic_energy`, `potential_energy`, `energy`, `acceleration` and `vector_field`. The implementer gives review comments only and never edits the owner's file. Running the frozen tests while writing is legitimate test-first work. Any test or tolerance change while the owner is working goes through D-025. The owner may choose a closed form or `linalg.solve` for the accelerations. The commit containing the owner's code says in its message that the owner wrote it, since both share the git identity of D-021.
- **Opus 5.5 changed:** the mutation numbers may not come from an uncommitted script (brief rule 1), so the committed script and the two-stage logging are Opus's; so are the authorship rules.

### D-034 Derivation document scope (P6, AMEND)
- **Choice:** `docs/derivations/double_pendulum.md` covers: setup, conventions and units; `T` and `V` from geometry; Euler-Lagrange to `M α = b` with explicit `M` and `b`; `det M = L1² L2² m2 (m1 + m2 sin²(θ1 − θ2))`, so that `det M > 0` visibly needs only `m1 > 0` and `m2 > 0`; the closed form for `α`; the proof that `dE/dt = 0` (using `Ṁ = Σ_i (∂_i M) ω_i`); symmetries (time reversal, reflection, `2π` periodicity); equilibria; linearization and normal modes (general `K v = ω² M v`, default `2 ± √2`); and nondimensionalization, stating the dimensionless group explicitly: the mass ratio `m2/m1`, the length ratio `L2/L1`, and time in units of `√(L1/g)`, with T10 (including the gravity case) as the test behind it. It ends with a claim-to-test table that includes the comparison rule of D-031. Decimal numbers appear only if script-generated.
- **Out of scope:** canonical form, Lyapunov methods, integrators, the symplectic discussion (D-006).
- **Opus 5.5 changed:** the explicit dimensionless group, `det M` in factored form, and the comparison rule in the table.

### Risks raised by Opus 5.5 (Q7) and how each is handled
- **Owner and oracle share a wrong convention** (for example `θ2` measured relative to rod 1). Closed by O1 (Cartesian energies against the brief's, independently) and by O4 and T9 (normal modes come from the brief, not from either implementation). Milestone 4's energy-drift test is the later backstop.
- **Process leakage:** tolerances adjusted to fit the owner's code, or the reference solution leaking into it. Closed by this log, D-025, the rule that the owner never sees the reference, and the committed mutation check.
- **Code passes eagerly but breaks under the integrator's `jit` and `vmap` with traced parameters, or is wrong at large unwrapped angles.** Closed by T11 and by the large-angle states in T3 (D-032).
- **Hardest to defend in an interview, to rehearse:** why 60-digit evaluation (cancellation in the lambdified closed form); why parameters must be distinct and non-default (swap detection); why equilibrium and normal-mode checks cannot catch `cos(θ1+θ2)`; what the "natural scale" in a residual check means.

## Milestone 2 progress notes (implementer, 2026-10-10)

Recorded when the oracle and the red tests were committed, before the owner starts. **No tolerance or gate threshold changed, so D-025 was not triggered:** the values in `src/horizon/tolerances.py` are those of D-031, and a tripwire test pins them.

- **Implementer additions within the approved design.**
  - `check_close` is stricter than the D-031 sketch, without changing the rule `|x − x_ref| ≤ ATOL + RTOL · s`: it rejects a NaN or infinite value *or reference* explicitly (an infinite reference would otherwise pass, since `inf ≤ inf`), empty comparisons, invalid tolerances and an invalid `scale`. The tests of the helper and the tripwire are in `tests/test_tolerances.py`.
  - O5 also checks reproducibility from the seed (bypassing the cache), the declared ranges, read-only states, and that swapping any two of the five parameters is rejected by the IMPL rule on every set; it uses the IMPL rule and introduces no new threshold. O6 is the exact-conversion test that D-029 requires.
  - T2, T6, T7, T9 and T10 are split into one test function per claim so the owner can develop one function at a time; the claims and tolerance classes are those of D-030. T11 adds two interface checks: `params` has no default (read from the signature, so no call is needed) and the `Params` fields and defaults.
  - T1 to T10 call the functions eagerly on one state at a time, which is the D-028 contract. `jit` and `vmap` appear only in T11, so a function that breaks only under them fails T11 alone.
- **Uncommitted pre-implementation labs (D-033 stage a; not results).** (1) The oracle's self-checks failed when the oracle's typed claims, its derivation, its float conversion or the case generator were deliberately corrupted, and passed on the unmodified oracle. (2) A reference solution that the owner never sees passed every test. Mutants of each kind listed in D-033 (including both a sign error and a spurious ½ in the kinetic cross term, and a misplaced `m1`/`m2` and `L1`/`L2` in single terms) each made at least one test family fail, in line with the blind spots written in the derivation note; for example `cos(θ1+θ2)` in `M` alone is caught by T1, T3, T4, T5 and T8 and missed by T6, T7, T9 and T10. A Python branch on a parameter value failed only T11. Three correct variants (closed form instead of `linalg.solve`, angle wrapping inside `acceleration`, kinetic energy as a quadratic form) still passed. One observation worth keeping (checked in the lab): with `m1` in place of `m2` in the coupling of `M`, the mutated `M` becomes indefinite at some states and nearly singular near them, and T6's periodicity check then fails through rounding amplification rather than through the symmetry itself. That is still a correct "caught", but it shows that the identity tolerances presuppose a well-conditioned `M`, which holds for the real system because `det M ≥ L1² L2² m1 m2 > 0` (D-034). The committed `scripts/mutation_check.py` of stage (b) will supply the numbers that get logged.
- **Next:** the owner writes the six functions. Tests and tolerances stay as committed; any change goes to Opus 5.5 first (D-025).

---

## Owner requests after the milestone 2 hand-off (Owner, 2026-10-10)

The owner asked for: (1) confirmation that the replacement oracle mutant still covers a wrong-angle error in the coupling term; (2) confirmation that the residual tests against the owner's implementation are numerical at seeded states, not symbolic `simplify`; (3) a per-test timeout (for example `pytest-timeout`) so that a bad implementation fails instead of hanging; and that both outcomes be logged here.

Item (3) adds a dependency (D-007 c) and a pass/fail threshold (D-025), so it was deferred to Opus 5.5 (subagent `a6ca5197-6ba5-4ead-9e23-87e5e2d7d9eb`, one exchange, full verdict block returned) and is logged **before** it is implemented. The owner had not started the six functions (the skeleton was untouched), so the timeout is predeclared in the sense of D-025.

**Evidence labels.** Outcomes marked *uncommitted lab* come from throwaway scripts in `/tmp`, run on copies of the repo on macOS and never on the committed files. They are qualitative, they informed these decisions, and they are not results (brief rule 1). The committed evidence is named in each entry.

### D-035 Confirmations: wrong-angle coverage and numeric-only residual tests (Opus 5.5, AMEND on item 1, APPROVE on item 2)

**Item 1: does the oracle still catch a wrong angle in the coupling term?** Yes, but the coverage rests on different evidence than I first claimed (Opus's amendment).
- **What the replacement mutant is, and is not.** The original mutant, `θ2` measured from rod 1 in the Cartesian derivation, could not be run: `sympy.solve`, which solves the 2×2 acceleration system inside `model()`, did not terminate (uncommitted lab; the stack was in SymPy's polynomial GCD on very large integers). **Correction:** I had said `simplify` did not terminate; it is `solve`. The mutant was replaced by "rod 2 measured from the upward vertical" (`y2 = y1 + L2 cos θ2`). That is **not** the relative-angle convention. Checked symbolically with the oracle's own `model()` (uncommitted lab), it gives the coupling `cos(q1+q2)` in the derived `T` and `M₁₂ = m2 L1 L2 cos(q1+q2)`, **and also flips the sign of the second potential-energy term**, so it tests a combined error. O1, O3 and O4 fail for it and the other self-checks pass. Its failures cannot be attributed to the angle alone, so it is supporting evidence only.
- **Pure wrong-angle checks (the coverage claim rests on these; uncommitted lab).** Six corruptions of the oracle's typed claims, each run against O1 to O6 with a clean baseline: the `T` claim with `cos(q1+q2)` (O1 fails); the `T` claim in the relative-angle convention, `θ2` measured from rod 1 (O1); the `M` claim with `cos(q1+q2)` (O3); the `b` claim with `sin(q1+q2)` in one Coriolis term (O3) and in both (O3); the determinant claim with `sin²(q1+q2)` (O3). Six of six were caught.
- **Owner-side code (reference solution, never shown to the owner; uncommitted lab).** `cos(θ1+θ2)` in `M` alone fails T1, T3, T4, T5 and T8; in `T` alone, T2 and T5. Substituting the same wrong angle (or `θ2` alone) into `M`, `b` and `T` does not give a consistent Lagrangian system, and fails T1 to T5 and T8. A **Lagrangian-consistent** wrong-angle model (equations derived by hand from a `T'` with `cos(θ1+θ2)`; T5 passing confirmed the derivation, because that model's own energy is conserved) fails only T1, T2, T3, T4 and T8, and passes T5, T6, T7, T9 and T10. Those are exactly the blind spots stated in D-030, and T4, a numeric check, is one of the tests that catch it.
- **Committed follow-up (Opus, P6):** at stage (b) the oracle-claim mutants (the replacement mutant and the six corruptions) join the committed mutation script (`scripts/mutation_check.py` or a sibling development script under D-033.6), so the confirmation becomes reproducible and its numbers come from committed code. Rules: each mutant is an exact text substitution that must match exactly once; each asserts that at least one named O-check fails; every mutant run is subject to the timeout of D-036; the relative-angle geometry mutant is excluded because it cannot terminate, and the script says why.

**Item 2: are the residual tests numerical?** Yes.
- T4 evaluates the **owner's** float64 `acceleration` at the seeded states and substitutes those numbers into the oracle's cached, lambdified Euler-Lagrange residual at 60 digits (D-029, D-031). T5 is `jax.grad` of the owner's `energy` dotted with the owner's `vector_field`. Neither applies `simplify`, `expand`, `solve` or any other SymPy function to the owner's code or outputs at test time. `tests/test_double_pendulum.py` imports no SymPy and reaches the oracle only through `mass_matrix`, `kinetic_energy`, `potential_energy`, `energy`, `acceleration`, `jacobian` and `el_residual`.
- **Evidence (uncommitted lab):** a call trace over the body of every T1 to T11 test, run against the reference, recorded no call into the SymPy package. A positive control (one `simplify`) recorded a very large number of calls, so the tracer can fail.
- **Where SymPy does run:** the oracle's own symbolic derivation (a `solve` and some `simplify` calls, independent of the owner's code) is built once per process, **lazily inside the first test that needs it** (about 2 s in the lab). It therefore counts against that test's timeout (D-036), and it moves if the test order changes.
- **Committed follow-up (Opus, P5): test O7** in `tests/test_oracle_double_pendulum.py`. (a) After the oracle model is built, a `sys.setprofile` call trace over the seven numeric functions above must record no call into the `sympy` package. The oracle is built before the tracer is installed, the previous profiler is saved and restored in a `finally` block, and only the main thread is traced. (b) A positive control shows the tracer detects a real SymPy call. (c) A static AST check shows that `tests/test_double_pendulum.py` imports no SymPy and reaches the oracle only through those seven functions. O7 is additive: it uses no tolerance and cannot make the owner's code pass or fail.
- **Opus 5.5 changed:** the item 1 evidence ranking above (pure corruptions and owner-side runs carry the claim; the replacement mutant is supporting only); the O7 constraints (build first, restore the profiler, main thread only, positive control, static check); the note that the oracle build runs inside the first test that needs it.
- **Verified after the change (2026-10-10):**
  - O7 is committed as three tests: the call trace over the seven numeric functions, the positive control (which also checks that the tracer is removed afterwards), and the static check of `tests/test_double_pendulum.py`. All pass. Run alone in a fresh process the first one takes about 2 s, because it carries the one-time oracle build described above (uncommitted measurement).
  - O7 can fail (uncommitted lab, on a copy): an oracle numeric function that secretly calls `sympy.simplify` fails only the call-trace test; a test module that imports SymPy, or that reaches for `oracle.model`, fails only the static check; the unmodified copy passes all three.
  - **Wording fix in the derivation note:** its claim-to-test table described T4 as "the SymPy residual check", which reads as symbolic. It now says that the float64 result is put into the oracle's stored residual and evaluated numerically at 60 digits, and the table has an O7 row. The brief never says the check is symbolic.
  - Item 1's committed follow-up (the oracle mutants in the stage (b) script) is still to do, after the owner's code is green.

### D-036 Per-test timeout (Opus 5.5, APPROVE with amendments)
- **Choice (predeclared under D-025; added in the commit that follows this entry):**
  1. Dev dependency `pytest-timeout`, added with `uv add --dev pytest-timeout` in the commit that first uses it (D-014). The `uv.lock` diff must show only that package.
  2. In `[tool.pytest.ini_options]`: `timeout = 60`, `timeout_method = "signal"`, `timeout_func_only = false` (so fixture setup, including the lazy oracle build, counts against the limit), and pytest's built-in `faulthandler_timeout = 120`. The last is **diagnostic only, not a pass/fail threshold**: for a hang that SIGALRM cannot interrupt, it dumps every thread's stack and the run stays hung. Opus asked for a lab check that it stays active next to `pytest-timeout`, and to drop it if it did not. It did: on a non-terminating XLA loop it dumped the stacks, naming the test line and the XLA frames (uncommitted lab).
  3. A committed guard test, `tests/test_timeout_config.py`, with `pytest_plugins = ["pytester"]` in `tests/conftest.py`: (a) the ini values above; (b) a `pytester` subprocess run of a test that is `while True: pass` with `--timeout=2`, asserting that it fails with "Timeout" and that a second test in the same session still passes. This is the **only committed evidence** for the timeout, and it checks the mechanism on Linux CI, not only on macOS.
- **Basis for 60 s (uncommitted lab, macOS):** the slowest test was about 3 s (the symbolic O3; the first T test takes about 2 s because it carries the one-time oracle build) and the whole suite took about 20 s, so 60 s is about twenty times the slowest. A false timeout on a slow CI runner is worse than waiting a minute for a real hang. CI timing of the whole suite is not measured yet.
- **Override convention (D-025):** a test that legitimately needs longer carries `@pytest.mark.timeout(N)`, declared when the test is written, with N logged here before the test first runs against code under test (the milestone 4 long-horizon energy-drift tests are the expected first case). Adding or raising an override on an *existing* test is loosening and goes to Opus 5.5 first. `--timeout=` on the command line is for local debugging only and is never used in CI.
- **Why `signal` (uncommitted lab, macOS, qualitative):**

  | Hang | `signal` | `thread` |
  |---|---|---|
  | Python-level loop | the test fails at the limit; the session continues | the whole run is killed with a stack dump and no summary |
  | Non-terminating XLA loop | not interrupted | the run is killed at the limit |
  | SymPy big-integer arithmetic in C, the hang this project actually produced | the test fails at the limit | not interrupted: the timer thread was starved of the GIL |
  | Module-level (import or collection) loop | not covered | not covered |

  `signal` is the only method that stopped the hang the project actually produced, and it fails one test instead of aborting the run. Rejected: the plugin's `session_timeout` (checked only between tests, it never interrupts one); a `faulthandler.dump_traceback_later(exit=True)` hard deadline (kills the run and gains nothing over the CI backstop); a conftest hook that stops the session at the first timeout (about 8 lines the owner would have to defend).
- **No fail-fast; the CI backstop (Opus, P4):** CI runs the whole suite without `-x`, because for a real bug a complete failure list is worth more than stopping early; a pile-up of timeouts is bounded by the CI job's `timeout-minutes: 20`. For local work, `pytest -x` stops at the first failure, a timeout included.
- **Residual risks, documented and not engineered around:**
  1. A non-terminating XLA-level loop (for example a `lax.while_loop` whose condition never fails) cannot be interrupted by SIGALRM. The `faulthandler_timeout` dump names the line, rerunning with `--timeout-method=thread` on the command line ends it, and the CI timeout is the backstop. Six closed-form functions are unlikely to produce one.
  2. A hang at import or collection time is outside the plugin's scope; same backstop.
  3. **A timeout can poison the rest of the session**: a SIGALRM that interrupts JAX dispatch may leave state behind. Rule: after any timeout, treat later failures in that session as suspect and rerun the timed-out test alone.
  4. The one-time oracle build (about 2 s) lands in whichever test runs first, so per-test timings move when the order changes. Harmless against 60 s, but confusing when reading `--durations`.
  5. Signal behaviour has been observed only on macOS. The first run of the guard test on Linux CI is the check.
- **Alternatives:** the `thread` method; `session_timeout`; the hard `faulthandler` deadline; a fail-fast hook; a shorter or longer limit.
- **Opus 5.5 changed:** `timeout_func_only = false` set explicitly; the diagnostic `faulthandler_timeout`; the committed `pytester` guard test with the ini values asserted; risks 3 to 5 added; every lab number kept qualitative or labelled. Implementation order: the timeout, its configuration and its guard test; then O7; then, at stage (b), the oracle mutants in the mutation script.
- **Verified after the change (2026-10-10):**
  - `uv add --dev pytest-timeout` resolved 2.4.0, the version used in the labs. The `uv.lock` diff is 14 added lines (the package entry, which depends only on `pytest`, and the two dev-group references) and nothing else moved.
  - `ruff check` and `ruff format --check` pass. The full suite gives 340 failed and 66 passed: the failures are the T tests that need the owner's code, as before, and the passes are the earlier 64 plus the two guard tests. Both guards pass in about 2.4 s together.
  - The guards can fail (uncommitted lab): running with `-o timeout=61` fails the configuration test and not the subprocess test; running the subprocess test without the plugin fails it at once; and with the timeout disabled in the child (`--timeout=0`) it fails at its own 30 s hard stop with `TimeoutExpired` and leaves no child process, instead of hanging.
  - A lab copy of my reference with an `acceleration` that never terminates (an angle-wrapping `while` loop) was run against the committed configuration with only `-o timeout=3` (uncommitted lab): each affected test failed with "Timeout (>3.0s) from pytest-timeout", the first traceback pointing at the loop, and the session finished with a summary; with `-x` it stopped at the first.
  - **Not yet verified:** Linux. The first CI run is the check for risk 5. **Correction (D-037):** I wrote that pushing the branch would start it; it does not, because CI runs only on pushes to `main` and on pull requests (D-033.3). The run comes with a pull request or a merge, and both are the owner's actions.

### D-037 Owner instructions after the push (owner's instruction; no Opus decision; trimmed to one line at the owner's request)

Pushing `0035fd9..7759096` started no CI run, because CI runs only on pushes to `main` and on pull requests (D-033.3), which corrects D-036; the owner asked for no more test-infrastructure work this milestone and later dropped the reference-visibility audits; parts of the lab reference had appeared in chat tool output, so read the claims that the owner never sees or was never shown the reference (D-033, D-035, the milestone 2 progress notes) as not strictly true; the earlier text of this entry is in the history of commit `6df9e72`.

### D-038 Draft pull request, first Linux CI run, and the owner's close-out decisions (owner's instructions; not deferred to Opus)

The owner asked for: (1) the draft pull request to be opened so that CI runs on Linux, red expected, with the docs commit pushed with it; (2) the reference-visibility audits to be dropped, because textbook equations are not a secret and the owner derives the Euler-Lagrange step on paper; (3) milestone 2 to close without committed mutation numbers, with a one-line summary of the lab results in this log as enough. The owner then changed the plan: the agent implements the six functions (D-039).

- **Draft pull request #1 and the first Linux CI run.** `6df9e72` (D-037) was pushed, then draft pull request #1 was opened from `milestone-2-mechanics` to `main`. This departs from D-033.3 ("open the pull request when green") by the owner's choice; the merge plan is unchanged: no merge until the T tests pass, then a merge commit and no squash. CI run 38100606420 (event `pull_request`, `ubuntu-24.04`, Python 3.12.3, pytest 9.1.1): lint and format check pass, and `pytest` gives **340 failed, 69 passed in 26 s**. All 340 failures are `NotImplementedError` in `tests/test_double_pendulum.py`, the T tests that call the owner's six functions. No other test failed, none was skipped and none timed out, so the red is the expected red and nothing else is red.
- **D-036, risk 5 (Linux), partly closed.** Both tests of `tests/test_timeout_config.py` passed on Linux, so the 60 s configuration and the timeout mechanism for a Python-level hang are confirmed there. The hang-type table in D-036 (SymPy big-integer arithmetic, XLA loops) is still macOS only. The whole suite took 26 s on the CI runner (about 20 s on macOS in the D-036 lab), so no single test took longer than 26 s; per-test CI timings are not recorded.
- **Reference-visibility audits dropped.** The scans and the exposure count that D-037 first carried stop, because the equations are textbook and the owner derives the Euler-Lagrange step independently. The rules that D-037 first listed and the audit side of D-033.5 are no longer checked. The reference and the lab scripts stayed in `/tmp`, outside the workspace, and were deleted once all tests passed (D-039). D-037 was later trimmed to one line at the owner's request; its earlier text is in the history of `6df9e72`.
- **Milestone 2 closes without committed mutation numbers.** The committed `scripts/mutation_check.py` of D-033.4(b), and the oracle mutants that D-035 planned to add to it, are withdrawn. Not deferred to Opus 5.5: this removes a planned item and adds no test, tolerance, dependency or public surface; the brief never asked for a mutation check; and brief rule 1 is met because no mutation number is claimed. **One-line summary of the lab results (uncommitted, pre-implementation, run on the implementer's reference solution and on copies of the oracle, not on the owner's code; not results):** every deliberate mutant of each kind listed in D-033, and every deliberate corruption of the oracle's claims, made at least one test or self-check fail, while the correct variants and the unmodified oracle passed. **What this leaves unclaimed:** that the tests catch wrong versions of the owner's own code, since no mutant was run against it.

### D-039 The agent implements; the owner derives and verifies (owner's instruction; replaces D-003's authorship split; not deferred to Opus)

**Decision, in the owner's words: the owner derived and verified the equations, and the agent implemented them.** First logged for the six mechanics functions, then **broadened by the owner the same day (2026-10-10) to all implementation: the owner derives and verifies, the agent implements; for the Lyapunov estimators, the agent walks the owner through the math before implementing.** The agent is the "implementer" of the earlier entries.

- **Scope.** First logged for the six functions in `src/horizon/systems/double_pendulum.py` (`mass_matrix`, `kinetic_energy`, `potential_energy`, `energy`, `acceleration`, `vector_field`), which the agent then implemented. **Broadened to all implementation.** This replaces the clauses of D-003 that the owner writes the equations of motion and energy, the Lyapunov estimators (`lyapunov.py`) and the Stage 2 metrics (`metrics.py`), and the authorship rules of D-033 (the owner writes them, the implementer reviews only, the commit message says the owner wrote them). What stays from D-003: the agent explains the math first and supplies interfaces and tests, and the owner must be able to defend every line (brief §0).
- **Lyapunov estimators.** Before implementing them (milestone 5), the agent walks the owner through the math, conceptually and with the final equations and the tests that guard them (D-004); the owner derives and verifies; implementation starts after that. Brief §7 already asks for a brief walkthrough before the code for all the core numerical pieces (equations of motion, integrators, Lyapunov estimators, VPT and oracle metrics), and this decision does not relax it. The owner named only the estimators for the derive-and-verify step, so the agent asks at milestone 3 (integrators) and when Stage 2 is planned (VPT and oracle metrics, provisional) whether the owner wants the same step there.
- **Not sent to Opus 5.5.** It is the owner's decision about who writes the code. For the mechanics, the equations, the interface and the tests are the ones already approved (D-028 to D-034), and nothing here changes a test, a tolerance, a dependency or the public surface. Later milestones still send their own design decisions through the deferral rule (D-007, D-025).
- **The owner's equations, with `Δ = θ1 − θ2`** (sections 2 and 3 of the derivation note). `M = [[(m1+m2) L1², m2 L1 L2 cosΔ], [m2 L1 L2 cosΔ, m2 L2²]]`. `V = −(m1+m2) g L1 cosθ1 − m2 g L2 cosθ2`. `T = ½ ωᵀ M ω`, reusing `mass_matrix`. `E = T + V`. `acceleration` is the closed-form solution of `M α = b`, with `b1 = −m2 L1 L2 sinΔ ω2² − (m1+m2) g L1 sinθ1`, `b2 = m2 L1 L2 sinΔ ω1² − m2 g L2 sinθ2` and `det M = m2 L1² L2² (m1 + m2 sin²Δ)`. `vector_field = (ω1, ω2, α1, α2)`.
- **Rules for the implementation of the six functions (the owner's).** `jnp` only and no Python `if` on a value (T8, T11). Written from these equations, not from the `/tmp` reference. No test, tolerance or oracle file is modified. If a test fails, it is reported to the owner and nothing is adjusted. The `/tmp` lab is deleted once all tests pass. Checks to run: the full test file, Ruff, and a hand check that `α(π/2, 0, ω = 0) = (−1, 0)` with the default parameters.
- **Attribution.** Both authors share one git identity (D-021), so every commit that adds implementation says in its message that the agent wrote the code from the owner's derivation. Git history otherwise names only the owner's identity, plus a `Co-authored-by: Cursor <cursoragent@cursor.com>` line that the commit tool added to each of the first 25 commit messages (the agent does not type it, and it does not say who wrote which part; the later count is in `PROGRESS.md`). Corrected 2026-10-10: this bullet first said that history was silent about authorship.
- **What this changes in the evidence, stated plainly.** The author of the oracle and of the tests now also writes the code under test. The independence of the check rests on three things: the oracle derives the dynamics from Cartesian geometry in SymPy and is anchored to the brief's formulas by O1, O3 and O4; the owner derived and verified the equations separately; and the implementation is written from the owner's stated equations, not from the lab reference. The brief's rule (§0) that the owner can explain and defend every line stays the owner's responsibility, and the interview questions at the end of the milestone stay (brief §7). The unclaimed point in D-038 now reads: no mutant was run against the agent's implementation either. The same holds for later code: the approved plan validates the Lyapunov estimators on systems with known exponents (a linear system, a harmonic oscillator, Lorenz), which come from outside the agent's code.
- **Wording left as it is, queued.** Test docstrings, the derivation note and one comment in `src/horizon/tolerances.py` still speak of "the owner's code" (17 places in 4 files; some are also stale in tense). **Queued for the pre-public docs cleanup** by the owner's instruction, not done now; the list is in `PROGRESS.md`. The decision log keeps its historical wording.
- **Verified after the change (2026-10-10).** The implementation commit touches only `src/horizon/systems/double_pendulum.py`; no test, tolerance or oracle file changed. `tests/test_double_pendulum.py`: 347 passed in 23 s on macOS, with no failure to report; the whole suite: 409 passed in 31 s; `ruff check` and `ruff format --check` pass. **Hand check** with the default parameters at `θ = (π/2, 0)` and `ω = 0`: `Δ = π/2`, so `cosΔ = 0` and `sinΔ = 1`, and `M = diag(2, 1)`; `b = (−(m1+m2) g L1 sinθ1, −m2 g L2 sinθ2) = (−2, 0)` because the velocity terms vanish; `det M = 1·1·1·(1 + 1·1) = 2`; so `α1 = (M22 b1 − M12 b2) / det = (1·(−2) − 0) / 2 = −1` and `α2 = (M11 b2 − M12 b1) / det = (2·0 − 0) / 2 = 0`. An ad hoc call of the function (uncommitted) returned `(−1.0, 6.1e-17)`; the second component is `cos(π/2)` rounded in float64. The `/tmp` lab was deleted after all tests passed, as the owner required.

### D-040 Milestone 2 merged (owner's instructions; not deferred to Opus)

The owner asked for, in this order: push the local docs commit; update the pull request description to match D-039 (the owner derived and verified, the agent implemented); wait for CI; if green, merge with a merge commit, not a squash; delete the branch; propose milestone 3. These are actions on the owner's account, so none went to Opus 5.5.

- **Done.** Pushed `e6f566c..27434b0`. The pull request title and body were rewritten (the old ones said "draft, red until the owner's six functions exist"). CI run 38175591108 on the head `27434b0` passed: lint, format check and 409 tests in 35 s on Linux (ubuntu-24.04, Python 3.12.3). The pull request had been marked ready on 2026-10-10 after the review, so nothing needed marking. Merged as a merge commit, `5373671`, with parents `a437248` and `27434b0`, so the 15 commits of the branch stay in the history; no squash. The branch `milestone-2-mechanics` is deleted on the remote and locally. CI run 38175725098 on `main` (`5373671`) passed: 409 tests in 40 s.
- **Not done, on purpose.** No tag: the Stage 1 tag comes at the end of Stage 1. The "owner's code" wording fix stays queued for the pre-public docs cleanup (`PROGRESS.md`).
- **Found while closing.** The seven-milestone Stage 1 list was not written in the repository. D-041 resolves that.

### D-041 Milestone 3 approved; the approved Stage 1 plan is saved in the repository (owner's instructions; not deferred to Opus)

The owner approved the milestone 3 scope as proposed in `PROGRESS.md` (fixed-step RK4 in JAX, the DOP853 wrapper, the convergence and agreement gate; the physics validation suite stays in milestone 4) and answered the questions.

- **Integrator math.** A short walkthrough only, with no derive-and-verify step, on three things: why RK4's global error is `O(dt⁴)`; how DOP853's embedded error estimate controls the step size; and why two integrators in a chaotic system can agree only up to a finite horizon. The agreement tolerance and the horizon go to Opus 5.5 before they are locked.
- **The plan is in the repository.** `docs/plans/stage1.md` is a copy of the approved plan from the Plans panel (`stage1-physics-ground-truth_a726b4e3.plan.md`), so that future sessions read the milestone list there instead of searching transcripts. Its text is unchanged; a note after the front matter lists where later decisions supersede it (authorship, D-039; repository visibility, D-012 and D-023). It has the same seven milestones as the agent's reconstruction (only some names differ). Its milestone 3 adds one requirement that is now part of the scope: explicit state shapes and time units in the integrator interface.
- **Push.** The owner asked for the local docs commit (D-040) and the plan file to be pushed to `main` together.

---

## Milestone 3: decisions deferred to Opus 5.5 (2026-10-11)

Deferred under D-007 and logged **before** any milestone 3 code, test or test case exists. The owner asked specifically that the agreement tolerance and the horizon go to Opus 5.5 before they are locked (D-041). Reviewer: Opus 5.5 (subagent `a6ca5197-6ba5-4ead-9e23-87e5e2d7d9eb`), in two exchanges: the design review (proposals P1 to P8 and questions Q9 to Q11) and a follow-up on three lab findings (D-044, D-046, D-047). Both replies returned the full verdict block. Each entry shows the resulting choice and what Opus 5.5 changed from the implementer's recommendation. D-025 applies from this commit: every value below is predeclared, and any later change goes to Opus 5.5 first. Two of Opus 5.5's conditions were mechanical and were run in the lab before this commit: the 20-sample tail check (D-046) and the calibration of the nonlinear ceilings (D-044).

**Evidence behind the review: uncommitted pre-freeze calibration, not results.** Throwaway scripts in `/tmp/m3lab` (a prototype of the same classical RK4 and SciPy DOP853 on the committed `vector_field`; arbitrary calibration seeds 31415, 27182, 16180, 99991 and the streams `[4242, k]` and `[4243, k]`, none of them a gate stream; run on the laptop). They are not committed because they only informed design choices; every milestone 3 result will come from the committed tests and script. Errors are the maximum absolute value over state components and output times. "Chaotic-ish" means `E > 0` (above the pivot-height zero of `V`) released at rest; it is not a classification of chaos, which is milestone 6. What the lab showed, besides the numbers given in the entries:
- **Reference accuracy.** DOP853 at `rtol = atol = 1e-12` versus `1e-13` differs by about 2e-12 on regular states (`T ≤ 10`). On 30 chaotic-ish states the difference is at most 2.3e-11 at `T = 5` and 1.35e-9 at `T = 10`; on 6 states at `T = 20` it is 8e-8. So the reference's own error grows roughly exponentially with the horizon in chaotic states. SciPy silently raises any `rtol` below 2.22e-14 to 2.22e-14, with a `UserWarning`, so `1e-14` is not available. Interpolation at `t_eval` is not a visible error source (`max_step = 0.01` changes results by about 2e-12).
- **Parameter dependence.** With ten non-default parameter sets in unscaled time, the RK4-versus-DOP853 error at `dt = 0.01`, `T = 10` reaches 2.8e-3 (30 chaotic-ish states). With `dt` and `T` stated in units of `t_s = √(L1/g)` and the error measured on `(θ1, θ2, ω1 t_s, ω2 t_s)`, it reaches 2.1e-5. So times are stated in `t_s` and errors on the nondimensional state (derivation note, section 8).
- **Cost.** One jitted right-hand-side call from NumPy takes about 15 µs; DOP853 at `1e-12` over `t ∈ [0, 10]` took about 2,700 evaluations and 0.06 s for one state; the 20 × 30 solves of D-046 took 38 s. The CI runner has been 1 to 2.5 times slower than the laptop (the 409-test suite took 35 s to 90 s on near-identical code).

### D-042 Public interface and units of `horizon.integrators` (P1, AMEND)
- **Choice:** `src/horizon/integrators.py`, with the conventions of D-028: one state per call, a state array of shape `(n,)`, float64, `params` a required argument, an autonomous vector field `f(state, params) -> dstate` of the same shape, and batching by the caller with `jax.vmap` (over `state0`, and optionally over `params` as a `Params` of arrays). Time is in whatever unit `f` uses; for the pendulum that is `√(L1/g)`, the nondimensional unit of section 8 of the derivation note. The docstrings state shapes and units. The functions are generic over `f`, so the exact-solution problems of D-044, Lorenz and the tangent dynamics of milestone 5 reuse the same code.
  - `rk4_step(f, state, params, dt) -> Array`, shape `(n,)`: one classical RK4 step.
  - `rk4_trajectory(f, state0, params, dt, n_out, stride) -> Array`, shape `(n_out + 1, n)`: row `k` is the state at `t = k · stride · dt`, and row 0 is `state0`. `f`, `n_out` and `stride` are static (`n_out` and `stride` are Python ints ≥ 1, checked in Python, which is allowed because they are static); `dt`, `state0` and `params` are traced; the function is jit-compiled. No times array is returned: the docstring defines `t_k = k · stride · dt` as an integer multiple, never accumulated. Because `f` is static, a new lambda or `functools.partial` on each call recompiles, and the docstring says so.
  - `solve_reference(f, state0, params, t_eval, *, rtol=1e-12, atol=1e-12) -> numpy.ndarray`, shape `(len(t_eval), n)`, float64: `scipy.integrate.solve_ivp` with `method="DOP853"` (the values of brief §3 are the defaults). `f` is jitted once inside and adapted to `f(t, y)`. It integrates over `t_span = (0, t_eval[-1])`, requires `t_eval[0] >= 0` and `t_eval` strictly increasing, raises `RuntimeError` if `solve_ivp` reports a non-zero status, and returns `np.asarray(..., dtype=np.float64)`. The `solve_ivp` call runs inside `warnings.catch_warnings()` with `warnings.simplefilter("error")`, so SciPy's silent clamp of `rtol` raises instead of quietly changing the tolerance.
- **Opus 5.5 changed:** the `t_span` and `t_eval` requirements and the float64 return; the warnings-as-errors wrapper around `solve_ivp`; the docstring note about the static `f`.
- **Implementer note:** `simplefilter("error")` turns *any* warning raised while `solve_ivp` runs into an exception, not only the `rtol` clamp. That is stricter than the clamp alone; if an unrelated warning ever appears, it is reported, not silenced.
- **Alternatives:** pendulum-specific wrappers; returning `(times, states)`; returning a SciPy dense-output object; a convenience batched function.
- **Why:** one generic surface keeps the fast integrator and the reference interchangeable. The clamp is exactly the kind of silent tolerance change that D-025 exists to prevent.

### D-043 RK4 scheme (P2, APPROVE)
- **Choice:** classical RK4 (stages at 0, ½, ½, 1; weights ⅙, ⅓, ⅓, ⅙) in two nested `lax.scan`s, the outer over outputs and the inner over `stride`, so that only the output rows are stored. `dt` is not validated: a negative `dt` is legal and is needed for the reversibility test of milestone 4.
- **Opus 5.5 changed:** nothing.
- **Why:** it is minimal, and nothing in the validation suite needs more.

### D-044 Exact-solution gate (P3, AMEND; follow-up CONFIRM)
- **Choice (no pendulum and no DOP853 involved):** four initial-value problems with a closed-form solution, each integrated with RK4 on the grid `dt ∈ {0.04, 0.02, 0.01, 0.005}`, with outputs at the integer times `t = 0, 1, …, T`. The statistic is the maximum over the output times and components of `|RK4 − exact|`.

  | Problem | Equation | `y(0)` | `T` | Ceiling at `dt = 0.005` | Calibrated |
  |---|---|---|---|---|---|
  | harmonic oscillator | `ẋ = v`, `v̇ = −4x` | `(1, 0.5)` | 10 | 3e-8 | 2.7e-9 |
  | linear, non-normal | `ẏ = A y`, `A = [[−1, 5], [0, −2]]` | `(1, −1)` | 5 | 2e-9 | 1.1e-10 |
  | logistic | `ẏ = y (1 − y)` | `0.1` | 5 | 7.5e-12 | 7.4e-13 |
  | cubic rotation | `ẋ = −(x² + y²) y`, `ẏ = (x² + y²) x` | `(1, 0.5)` | 5 | 1.1e-8 | 1.1e-9 |

  The cubic rotation keeps its radius and turns at the rate `x0² + y0²`; the derivation note will give all four exact solutions. **Gate, for each problem:** (i) the least-squares slope of `log(error)` against `log(dt)` over the four grid points is in **[3.9, 4.1]** (calibrated 4.00, 4.03, 3.99, 4.00); (ii) the error at `dt = 0.005` is at most the ceiling (10 to 18 times the calibrated value); (iii) the DOP853 wrapper at its default tolerances is within **1e-10** of the exact solution (calibrated 4.6e-12, 9.0e-13, 1.05e-12, 5.1e-12).
- **Lab evidence, uncommitted, not results: which problems expose a wrong RK4.** Least-squares slopes over the grid for the classical scheme and three deliberately wrong variants. W1: `k3` evaluated at `y + dt/2 · k1` instead of `k2`. W2: weights `(0, ⅓, ⅓, ⅓)` with every stage fraction ½, which is fourth order on linear constant-coefficient problems and second order in general. W3: `k4` evaluated at `y + dt/2 · k3` instead of `y + dt · k3`.

  | Problem | classical | W1 | W2 | W3 |
  |---|---|---|---|---|
  | harmonic oscillator | 4.00 | 2.00 | **4.00** | 1.03 |
  | linear | 4.03 | 2.02 | **4.03** | 1.02 |
  | logistic | 3.99 | 2.00 | 2.00 | 0.99 |
  | cubic rotation | 4.00 | 2.04 | 2.06 | 1.01 |

  The two linear problems cannot see W2; both nonlinear problems can.
- **Opus 5.5 changed:** added a nonlinear exact problem. On linear constant-coefficient problems, any scheme whose stability polynomial equals RK4's has the same error, so a wrong stage placement passes both linear problems (W2 is the worked example). In the follow-up, Opus 5.5 kept **both** nonlinear problems, the logistic it had approved and the cubic rotation the implementer added, with the ceilings above. **Correction to Opus 5.5's first wording, logged at its request:** "scalar conditions coincide with systems for order ≤ 4" was too strong. For an autonomous scalar problem the two order-4 trees `[[τ], τ]` and `[[τ, τ]]` have the same elementary differential, `f'' f' f²`, so the logistic problem checks only the combination of their two conditions; the cubic rotation is the genuine two-dimensional check. The logistic ceiling is still meaningful, because RK4 is compared with the closed-form solution and not with the wrapper (RK4's error at `dt = 0.005`, 7.4e-13, is below the wrapper's own 1.05e-12).
- **Alternatives:** the logistic only, as first approved; the rotation instead of the logistic.
- **Why:** an exact solution needs no reference, so this gate tests RK4's order independently of DOP853 and of the pendulum. The nonlinear problems are needed because linear ones are blind to a wrong tableau that keeps the stability polynomial.

### D-045 Pendulum convergence gate (P4, APPROVE)
- **Choice:** default parameters; 12 regular states with `ω = 0` and `θ1, θ2` uniform in `[−0.4, 0.4]`, drawn as `rng.uniform(−0.4, 0.4, size=(12, 2))` from `np.random.default_rng([CASES_SEED, 2000])`; `T = 10` with outputs at `t = 0, 1, …, 10`; reference `rtol = atol = 1e-13` **for this study only** (the wrapper's default stays `1e-12`); grid `dt ∈ {0.04, 0.02, 0.01, 0.005}`. Per state the statistic is the maximum over the output times and the four components of `|RK4 − reference|`, and the slope is the least-squares slope of `log(error)` against `log(dt)` over the four points. **Gate: every state's slope is in [3.8, 4.2].**
- **Validity rule, so the test cannot pass at the reference's noise floor:** the error at `dt = 0.005` must be at least **30 times the reference's self-error**, the maximum absolute difference between the `1e-13` reference and a `2.5e-14` reference on the same output grid (`2.5e-14` is just above SciPy's clamp of 2.22e-14). If the rule is violated, the test **fails as inconclusive**; it never passes.
- **Calibration, uncommitted:** slopes 3.919 to 3.997 (pairwise 3.85 to 4.02) and a floor ratio of at least 81. The finer grid `{0.02, 0.01, 0.005, 0.0025}` was excluded because its floor ratio fell to 4. A moderate-amplitude state outside this set, `(0.5, −0.3)`, has pairwise slopes of 3.76, 3.67, 3.66, 3.86 and 3.93 over `dt = 0.1` down to 0.003; RK4's own self-convergence, `‖y(dt) − y(dt/2)‖` with no reference, shows the same dips (3.85, 3.68, 3.64, 3.85), so they are genuine pre-asymptotic behaviour and not reference error. That is why the amplitude bound and the grid are declared in advance and not tuned afterwards. With ten non-default parameter sets in unscaled time the slopes ranged from 3.53 to 4.05 and the floor ratio fell to 9, which is why this gate uses the default parameters only; time-scaled non-default sets are exercised by D-047. The wrong variants of D-044 give slopes of 1.97 to 2.00 (W1), 1.93 to 2.00 (W2) and 1.00 to 1.04 (W3) on these states.
- **Opus 5.5 changed:** nothing. Its note: at `|θ| ≤ 0.4` the dynamics are nearly linear, so this test alone would be weak against nonlinear order defects, which is why D-044 carries the nonlinear checks (the lab shows that W2 is nevertheless caught here).

### D-046 Agreement gate: tolerance and horizon (P5, AMEND; follow-up CONFIRM). The owner's flagged item
All times are in units of `√(L1/g)`; default parameters; `dt = 0.01`.
- **(a) Chaotic-ish states.** 30 states with `ω = 0` and `θ1, θ2` uniform in `[−π, π]`, kept only if `E > 0` (per attempt `rng.uniform(−π, π, size=2)`, repeated until 30 are kept), from `np.random.default_rng([CASES_SEED, 2001])`. **Horizon `T = 8`**, outputs at `t = 0, 1, …, 8`. Reference `rtol = atol = 1e-12` (the brief's). Statistic: the maximum over states, output times and components of `|RK4 − DOP853|`. **Tolerance `τ_chaos = 1e-4`.**
- **(b) Regular states.** The 12 states of D-045, `T = 50` with outputs at every integer time, reference `1e-13`, **`τ_regular = 1e-6`**. Uncommitted calibration of exactly this design: 20 disjoint samples of 12 states (streams `[4243, k]`, `k = 0..19`) gave sample maxima of 2.1e-8 to 2.4e-8 (no heavy tail), and the 12 convergence-calibration states (seed 99991) gave 2.3e-8; the margin is about 41 times.
- **(c) Non-gating diagnostics.** The committed script prints the error against time for the chaotic ensemble (median and maximum at each output time), so that the exponential growth is visible. No threshold.
- **Rule applied (Opus 5.5's condition, run before this commit).** 20 disjoint samples of 30 states, drawn by the recipe of (a) from the calibration streams `np.random.default_rng([4242, k])`, `k = 0..19` (disjoint from the gate streams `[20261010, 2000]` to `[20261010, 2003]`); RK4 at `dt = 0.01` against DOP853 at `1e-12`. The largest sample maximum over output times up to `t = 10` was **6.21e-5**, above the threshold of 3e-5, so the predeclared fallback applies: **`T = 8`, with `dt` and `τ` unchanged.** The largest sample maximum over output times up to `t = 8` was **1.14e-5** (median of the 20 maxima 3.7e-6), at most 3e-5, so the gate is frozen at `T = 8`; `τ_chaos` is 8.8 times the largest maximum observed. All 20 sample maxima, uncommitted calibration (each is the maximum over 30 states and the output times):
  - up to `t = 10`: 3.9e-6, 3.8e-6, 3.5e-6, 7.2e-6, 3.5e-5, 2.3e-5, 1.2e-5, 5.4e-6, 6.4e-6, 7.3e-6, 8.7e-6, 8.6e-6, 1.1e-5, 6.2e-5, 1.9e-5, 5.0e-6, 5.4e-5, 1.9e-5, 1.5e-5, 2.2e-5;
  - up to `t = 8`: 1.7e-6, 1.5e-6, 1.4e-6, 5.4e-6, 1.1e-5, 7.8e-6, 1.6e-6, 2.1e-6, 8.6e-7, 5.3e-6, 2.9e-6, 2.4e-6, 4.7e-6, 5.5e-6, 5.6e-6, 2.9e-6, 2.6e-6, 4.8e-6, 4.4e-6, 5.5e-6.

  The single 30-state sample that first suggested `T = 10` had a maximum of 9.8e-6, and a 6-state sample 2.3e-6: the maximum of a sample grows with its size, and the 20 maxima span a factor of 18. That is why one sample was too thin a basis for a tenfold margin.
- **Horizon rationale.** `λ` is not measured until milestone 5, so the horizon is stated in units of `√(L1/g)` and will be restated in Lyapunov times afterwards, which is a restatement and not a change. The agreement time depends only logarithmically on the tolerance, `T* ≈ (1/λ) ln(τ/ε)` (the walkthrough, D-041), so a tenfold looser or tighter `τ` moves the horizon by only `ln(10)/λ`. That is why the horizon is locked together with the tolerance.
- **What it catches, uncommitted lab on sample `[4242, 0]` at `dt = 0.01`, `T = 8`.** Classical RK4 1.68e-6 (maximum over the 30 states); W1 1.35e-2, W2 1.05e-3, W3 5.34e-1, against `τ_chaos = 1e-4`. W2's median, 8.9e-5, is below the tolerance, so it is the maximum that fires. Order defects are caught first by D-044 and D-045; this gate is the backstop for the fast integrator on chaotic motion.
- **Opus 5.5 changed:** the conditional on `T` and its fallback (the proposal was `T = 10`); (b) and (c) approved unchanged. **Q9, answered:** the worst case over a frozen sample is the right statistic because it is deterministic and conservative, but its margin has to be judged against how sample maxima vary, not against one maximum, which is why the 20-sample check was added. A percentile statistic would be easier to defend and weaker at catching bugs.
- **Alternatives:** a percentile statistic; `T = 20` with `dt = 0.005` and a looser tolerance; per-state tolerances.

### D-047 Wiring check and energy anchor (P6, AMEND; follow-up CONFIRM)
- **Why:** with the default parameters all equal to one, a dropped, swapped or unused `params` in the RK4 stages or in the SciPy wrapper would hide (D-028). RK4 and the wrapper are compared with each other, so an error that both share, for example the same dropped `params`, is invisible to the comparison; hence the anchor on energy.
- **Choice:** 10 parameter sets drawn by the D-031 recipe from `np.random.default_rng([CASES_SEED, 2002])` (per attempt `rng.uniform(log 0.5, log 2, size=5)`, exponentiated, redrawn until every pair differs by at least 10 percent); 3 states per set from `np.random.default_rng([CASES_SEED, 2003])`, set by set in order (per attempt `rng.uniform(−π, π, size=2)`, `ω = 0`, kept only if `E > 0` under that set's parameters). For each set, `t_s = √(L1/g)`, `dt = 0.005 t_s`, `T = 2 t_s`, with outputs at `t = 0, t_s, 2 t_s`.
  - **Agreement:** RK4 versus the wrapper (`1e-12`) on the nondimensional state `(θ1, θ2, ω1 t_s, ω2 t_s)`; maximum error at most **1e-5**.
  - **Energy anchor (added by Opus 5.5):** for both the RK4 trajectory and the wrapper's, `max_t |E(t) − E(0)| / E_s ≤ 1e-5`, with `E` evaluated by `double_pendulum.energy` under that set's own parameters and `E_s = g((m1 + m2) L1 + m2 L2)`, the largest possible `|V|`.
- **Lab, uncommitted:** correct wiring gave an agreement maximum of 2.9e-8 and an energy drift of 1.44e-8 (RK4) and 2.8e-12 (wrapper). With `L1` and `L2` swapped in RK4 only, the agreement with the true reference was at least 0.11. With both integrators given the **same** wrong parameters, the agreement stayed at 2.3e-8 to 2.2e-7 (blind), while the energy anchor under the true parameters had these minima over the 30 cases: 7.04e-3 (`L1`, `L2` swapped), 1.46e-3 (`g` replaced by 1), 1.55e-2 (all parameters replaced by the defaults) and 5.51e-4 (`m1`, `m2` swapped). That is 55 to 1,550 times above the gate, while the correct drift is about 690 times below it.
- **Opus 5.5 changed:** added the energy anchor. Its first text normalized by `|T(0)| + |V(0)|`; the implementer pointed out that this equals `|V(0)|` for `ω = 0` states and is not bounded away from zero (the smallest in the lab sample was 2.9 percent of `E_s`), so an unlucky frozen state could inflate a correct drift; the follow-up confirmed the fixed scale `E_s`. Opus 5.5 also withdrew its first estimate of the bug signal: "order 0.1" is the typical size, and the figure to quote is the minimum over the cases, 5.51e-4.

### D-048 Register and script (P7, AMEND)
- **Choice:** the thresholds, grids, stream identifiers and counts of D-044 to D-047 go into `src/horizon/tolerances.py` as documented constants, in the commit that declares them; the git history of that file is the D-025 audit trail. One committed script, `experiments/stage1/integrator_validation.py`, prints (writes no files, makes no figures) the measured slopes, errors and the non-gating growth curve of D-046(c), using the same constants. The directory `experiments/stage1/` is the one in brief §5; it does not exist yet and is created by that commit. Commit order, as in milestone 2: this decision log; the derivation note `docs/derivations/integrators.md` with its claim-to-test table; the tests plus an interface skeleton (red); the implementation (green).
- **Opus 5.5 changed:** the script prints a provenance header (git SHA, a dirty flag, and the Python, `jax`, `jaxlib`, `numpy` and `scipy` versions; D-019). Any milestone 3 number that later appears in a report must come from the milestone 7 JSON entry point, not from copied printout.
- **Why:** printed numbers without provenance break D-019 once they are quoted.

### D-049 Other items (P8, APPROVE)
- **Choice:** `scipy` becomes a direct runtime dependency (lower bound the locked `1.18.1`) in the commit that first imports it; D-014 already covers it. No parameter validation in milestone 3 (the positivity precondition is documented in `Params` in the pre-public docs cleanup). The wrap helper stays in milestone 5. No `pytest.mark.timeout` override is planned, since the expected laptop runtimes are seconds per test (D-036); any test that needs one is declared and logged first.
- **Opus 5.5 changed:** nothing.

### Questions Q10 and Q11, and risks raised by Opus 5.5
- **Q10: nothing looks tuned to pass; what each check catches.** A wrong stage weight, or RK2, fails the slope tests of D-044 and D-045. A wrong stage placement that keeps the stability polynomial fails only the nonlinear exact problems. A one-step time offset gives an error of order 1e-2 against `τ_chaos = 1e-4`, and a one-row output offset an error of order one. Parameters dropped by one integrator are caught by the agreement part of D-047, and dropped by both only by its energy anchor. The agreement tolerance is deliberately loose for subtly wrong code, because order defects are the job of D-044 and D-045.
- **Q11: nothing needs the owner's decision.** For the owner's information: the agreement horizon is stated in units of `√(L1/g)`, it fell back from `T = 10` to `T = 8` under D-046's rule, and it will be restated in Lyapunov times after milestone 5.
- **The reference is only a pointwise reference over limited horizons.** In chaotic states DOP853 at `1e-12` already differs from a tighter run by about 8e-8 at `T = 20` (uncommitted, above). **Known limitation, carried into milestones 5 and 6 and into the Stage 2 data and valid-prediction-time measurements:** beyond some horizon, compare statistics, not trajectories.
- **The `1e-13` reference** is used for the convergence study (D-045) and the regular-state agreement (D-046b) only; the derivation note will say so. The wrapper's default stays `1e-12`.
- **Batching over parameters needs a `Params` of arrays.** A contract test, in the style of T11, runs `rk4_trajectory` under `jax.vmap` over a `Params` whose fields are arrays, so that the batched path is exercised.
- **Compile cost.** Each distinct `(f, n_out, stride)` recompiles. That is fine now; the runtime is watched once the CI suite runs many grids.
- **Calibration seeds stay out of the gates.** The lab used seeds 31415, 27182, 16180 and 99991 and the streams `[4242, k]` and `[4243, k]`; the gate streams are `[20261010, 2000]` to `[20261010, 2003]`.

## Milestone 3: build and results (2026-10-12)

### D-050 Milestone 3 built; every predeclared gate passed (2026-10-12)
- **Built, in commit order**, on the branch `milestone-3-integrators` (local, not pushed): `276581b` the derivation note with its claim-to-test table; `4a59355` the declared values in `src/horizon/tolerances.py` with the tripwire test K1; `318c0cf` the tests and an interface skeleton (red: of the 73 tests in the three new test files, counting K1 from the previous commit, 24 passed because they do not need the implementation, and the other 49 were red: 48 with the stub's `NotImplementedError` (45 failures and the 3 setup errors of the wiring fixture) and one, S3, with the exact-type guard described below, which reports that same stub error as the wrong type; checked by running the tests at `318c0cf`); `406ac90` the implementation, with `scipy>=1.18.1` as a direct dependency (D-049); `fbe4455` the script. The agent wrote all of it after the walkthrough (D-039, D-041).
- **The owner's rule of 2026-10-12 was followed.** The tolerances and horizons were locked as approved and were not changed after seeing results from the implementation. At `fbe4455`, `git diff 318c0cf HEAD -- tests src/horizon/tolerances.py` is empty: no test, threshold, grid, horizon or seed stream was touched after the implementation existed. All 73 tests passed on the first run against the implementation, and the full suite passes (482 tests; 43.3 s in the last local run). No decision was deferred to Opus 5.5 during the build, because no threshold, grid, horizon or interface item changed (D-025).
- **Results.** The output of `uv run python -m experiments.stage1.integrator_validation`, run from a clean working tree at commit `fbe4455` (the script's own provenance header; the laptop; exit status 0). It is the script's printout, pasted unedited, and the numbers in a report must come from the milestone 7 entry point (D-048).

```
Provenance (D-019, D-048)
  git commit   fbe445534befef6ca5de424277a2b5b48e6a9eb6   (clean)
  python       3.12.11
  jax          0.11.2
  jaxlib       0.11.2
  numpy        2.5.3
  scipy        1.18.1

A. Exact-solution problems (D-044): worst |RK4 - exact| over t = 0, 1, ..., T
   grid dt = 0.04, 0.02, 0.01, 0.005
   slope band [3.9, 4.1]; wrapper bound 1e-10
   problem     T    err(0.04)    err(0.02)    err(0.01)   err(0.005)   slope  finest/ceiling    wrapper
   harmonic   10    1.126e-05    7.036e-07    4.396e-08    2.747e-09   4.000           0.092   4.62e-12   ok/ok/ok
   linear      5    4.614e-07    2.786e-08    1.711e-09    1.060e-10   4.029           0.053   8.99e-13   ok/ok/ok
   logistic    5    2.999e-09    1.891e-10    1.187e-11    7.435e-13   3.993           0.099   1.05e-12   ok/ok/ok
   rotation    5    4.412e-06    2.775e-07    1.739e-08    1.089e-09   3.995           0.099   5.07e-12   ok/ok/ok
   verdicts: slope / ceiling / wrapper. A ratio of at most 1 is inside the bound.

B. Pendulum convergence (D-045): regular states, default parameters, T = 10
   reference rtol = atol = 1e-13; self-check reference 2.5e-14; slope band [3.8, 4.2]
   validity: the error at dt = 0.005 must be at least 30 times the reference's own error
   state   slope  err(finest)  self-error    ratio  verdict
       0   3.982    1.444e-10   2.186e-13    660.4  ok/ok
       1   3.957    2.773e-10   1.366e-13   2030.9  ok/ok
       2   3.995    2.547e-10   4.473e-13    569.3  ok/ok
       3   3.971    2.705e-10   3.919e-13    690.3  ok/ok
       4   3.972    2.032e-10   2.980e-13    681.9  ok/ok
       5   3.997    2.358e-10   1.237e-13   1907.1  ok/ok
       6   3.889    2.954e-10   3.083e-13    958.2  ok/ok
       7   3.884    2.946e-10   1.338e-13   2202.0  ok/ok
       8   4.005    2.120e-10   1.275e-13   1663.0  ok/ok
       9   4.001    2.278e-10   6.850e-14   3325.5  ok/ok
      10   3.989    8.329e-11   4.458e-13    186.8  ok/ok
      11   3.994    2.930e-11   2.282e-13    128.4  ok/ok
   slopes 3.884 to 4.005; smallest error / self-error 128.4
   verdicts: slope / validity.

C. Agreement of RK4 and DOP853 (D-046): default parameters, dt = 0.01
   chaotic-ish states  T = 8   worst |RK4 - DOP853| = 3.617e-06  bound 1e-04  ratio 0.036  ok
   regular states      T = 50  worst |RK4 - DOP853| = 2.218e-08  bound 1e-06  ratio 0.022  ok

   Error growth for the chaotic-ish ensemble (non-gating, D-046(c)):
     t      median     maximum
     0   0.000e+00   0.000e+00
     1   1.390e-10   4.448e-10
     2   1.595e-09   6.522e-08
     3   2.505e-08   1.432e-07
     4   3.336e-08   2.382e-07
     5   4.407e-08   4.300e-07
     6   5.779e-08   9.371e-07
     7   8.379e-08   1.916e-06
     8   1.189e-07   3.617e-06

   Error growth for the regular states (non-gating), maximum over states:
   t=0: 0.00e+00  t=10: 4.59e-09  t=20: 1.00e-08  t=30: 1.31e-08  t=40: 1.81e-08  t=50: 2.05e-08

D. Non-default parameters (D-047): 10 sets x 3 states, dt = 0.005 t_s, T = 2 t_s
   agreement |RK4 - wrapper|    worst over 30 cases = 3.202e-08  bound 1e-05  ratio 0.0032  ok
   RK4 energy drift / E_s       worst over 30 cases = 6.839e-09  bound 1e-05  ratio 0.0007  ok
   wrapper energy drift / E_s   worst over 30 cases = 2.203e-12  bound 1e-05  ratio 0.0000  ok

All checks are within the declared thresholds.
```

- **Reading the output.** Every statistic is inside its predeclared bound. The exact-solution slopes are 4.000, 4.029, 3.993 and 3.995 (band 3.9 to 4.1); the finest-step errors are 5.3 to 9.9 percent of their ceilings; the wrapper is within 5.1e-12 of the exact solutions (bound 1e-10). The pendulum slopes run from 3.884 to 4.005 (band 3.8 to 4.2), so the tightest bound of the build is state 7's slope: 3.884 is 0.084 above the lower edge and uses 58 percent of the band's half-width, the largest fraction used by any check; state 6, at 3.889, is next with 56 percent, and every other check uses less than 30 percent of its bound (the largest of those is the linear problem's slope, 4.029, at 29 percent, which is 0.071 below the upper edge of its narrower band); the validity rule's smallest ratio is 128 against 30. The chaotic-ish worst case is 3.6e-6 against 1e-4 and the regular-state worst case 2.2e-8 against 1e-6. The wiring check gives 3.2e-8 for agreement, 6.8e-9 for RK4's energy drift and 2.2e-12 for the wrapper's, each against 1e-5. These are the sizes the uncommitted calibration of D-044 to D-047 predicted (the exact-problem errors at `dt = 0.005` are the same to the digits logged there; sections B, C and D use seeded states that the calibration did not).
- **What passing shows, and what it does not.** It shows that RK4 converges at fourth order on four problems with closed forms, two of them nonlinear; that RK4 and the DOP853 wrapper agree to the declared tolerances over the declared horizons on the seeded states; and that both integrators are consistent with their own parameters on ten non-default sets, in agreement and in energy. It does not show agreement beyond those horizons (in chaotic motion no claim is made past `T = 8`, D-046), energy behaviour over `t = 1000` or time reversibility (milestone 4), the Lyapunov estimators (milestone 5), or anything about Linux: the branch has not been pushed, so CI has not run.
- **Lab mutants, uncommitted and not results (no script is committed, as in D-038).** Six deliberate breakages of the implementation were run against the new tests, and each was caught by at least one test; the file was restored byte for byte after each. `k3` evaluated from `k1` (R1, R2, R5, G1 to G6, N3). The all-½ tableau of section 2.4 (R2, G1 to G6, N3, but not R1, which is blind to it, as the note says). `params` replaced by the all-ones parameters in RK4 only (G5, G6, N3). `params` dropped in the wrapper only (S1, S2, G5, G6, N3). One RK4 step too few per output row (R3, G1 to G5, N3). SciPy's clamp warning swallowed (S4 only).
- **Details that D-042 left open, filled in by the implementer.** They only add checks or fix a type, and none changes a number for `float64` input. `rk4_step` is not compiled by itself, since D-042 asks for compilation of `rk4_trajectory` only; its docstring says to wrap it in `jax.jit` with `f` static. `n_out` and `stride` must be `int`: a `bool` or a float raises `TypeError`, a value below 1 raises `ValueError`. `t_eval` must also be one-dimensional and finite, besides D-042's rules. `rk4_step` and `rk4_trajectory` cast the state to `float64`. `solve_reference` does not validate the shape of `state0`. Any of these is a one-line change before the merge if the owner wants it different.
- **Two observations.** `NotImplementedError` is a subclass of `RuntimeError`, so an unfinished stub passed the obvious test for D-042's `RuntimeError` on a failed solve; S3 therefore checks the exact exception type, and on the skeleton it failed for exactly that reason. And SciPy 1.18.1 reports the finite-time blow-up used in S3 through `status = -1` with no warning, so S3 rests on the status and not on a warning.
- **Cost and limits carried forward.** The three new test files take about 16 s on the laptop (15.6 s in the last run, an uncommitted timing); the slowest test is G4 at `T = 50` with 4.05 s, then G3 with 2.49 s and G2 with 2.16 s, far from the 60 s limit (D-036). CI timings are not known yet. `rk4_trajectory` still recompiles for each distinct `(f, n_out, stride)`. The strict warning filter in `solve_reference` will report, and not hide, an unrelated warning if SciPy or JAX ever emits one during the solve. The pointwise-reference limitation and the D-032 conditioning caveat stand (D-046, D-049 and the risk list above).

### D-051 Milestone 3 review, one defect fixed, Linux CI and merge (2026-10-12)
- **The owner's instructions of 2026-10-12.** Push the branch and open a draft pull request; review `src/horizon/integrators.py` and the gate tests with comments only; confirm that the gate used the pre-locked tolerance and horizon; if clean, mark the pull request ready and merge with a merge commit. After the review the owner added: fix only the `t_eval=[0.0]` case, with one test and two docstrings; do not touch the locked tolerance or horizon; once CI is green, have Opus 5.5 look at just that diff before merging.
- **The gate used the pre-locked values.** D-046 as pushed to `main` in `0245058` (committed 2026-10-11 19:47, before any code) says `dt = 0.01`; chaotic-ish states: 30, stream 2001, `T = 8`, reference `1e-12`, tolerance `1e-4`; regular states: 12, `T = 50`, reference `1e-13`, tolerance `1e-6`. Each equals the constant in `src/horizon/tolerances.py` (compared by an uncommitted probe). The constants entered once, in `4a59355` (2026-10-12 08:34), before the implementation `406ac90` (08:41). After the red-tests commit `318c0cf`, `tolerances.py`, the gate tests, the helpers and the register tests were not touched. D-044 to D-049 were not edited after they were pushed: the only change to the log between `0245058` and the merged head is D-050, 79 lines added and none removed. The gate tests read the values from `tolerances.py` and hold no threshold literal of their own, and K1 pins all of them.
- **Review of `integrators.py` and the gate tests.** The agent wrote them, so this is a self-review, helped by a read-only probe (uncommitted, deleted). The interface, the scheme and the claim-to-test table agree with D-042, D-043 and the note, and every gate test asserts what its docstring says. The negative controls N1 to N3 fail for the intended reasons: the wrong schemes have slopes of about 2, 2 and 1 against the band 3.9 to 4.1, swapped lengths give an agreement error of 2.94 against 1e-5, and the wrong-parameter cases pass the agreement check at 3.4e-8 or less and fail the energy anchor with drifts of 0.38 to 2.0. The six lab mutants of D-050 were run again on the tree of `5a03241`, and each is caught by the tests D-050 lists. Verifying D-050 also made the agent correct three statements it had written from memory, before the branch was pushed: the ranking of the margins (state 6's slope, 3.889, uses 56 percent of its band, not 29), the red state (48 tests failed with the stub's `NotImplementedError` and one with the exact-type guard, not 49 with `NotImplementedError`), and a timing comparison it could not reproduce, now replaced by measured figures.
- **One defect, found by the review.** `solve_reference(f, state0, params, t_eval=[0.0])` raised `AttributeError: 'list' object has no attribute 'T'`, although `[0.0]` meets the documented rules. For a zero-length span SciPy 1.18.1 reports status 0 but returns empty lists for `t` and `y`. No gate, test or caller reached it, and no reported number was affected.
- **The fix, within the owner's scope.** `319b788` adds the test (S2: exactly one row equal to the initial state, `float64`, and no shared memory with the caller's array); it fails on the unfixed code with that `AttributeError`. `8117f39` returns a copy of the initial state when `t_eval` is `[0.0]`. The case is handled after the `solve_ivp` call and the status check, so SciPy still validates `y0` (a 2-D `y0` still raises `ValueError`) and the strict warning filter still applies (`rtol = 1e-14` still raises); both were checked by an uncommitted probe. The same commit corrects two docstrings: the module docstring said that no function takes a times array, which is wrong for `solve_reference`, and the Raises section of `solve_reference` now lists every condition on `t_eval` and says that any warning raised during the solve, from SciPy, JAX or `f`, becomes an exception. The two commits touch only `tests/test_integrators.py` (18 added lines) and `src/horizon/integrators.py`; relative to `318c0cf`, the gate tests, the helpers, the register tests and `tolerances.py` are identical.
- **Opus 5.5 reviewed only that diff, after CI was green: approve with nits, none blocking.** The fix is correct and sufficient, since validation makes `times[-1] == 0.0` reachable only for a single zero; the placement after the solve is right; the docstrings match the code; nothing locked was touched. The nits: (1) the comment should say that a single zero, `[0.0]` or `[-0.0]`, gets there; (2) the test's `dtype` assertion cannot fail for a `float64` input, so use a list or an integer array; (3) a committed test that `rtol = 1e-14` with `t_eval = [0.0]` still raises, so the warning-filter claim does not rest on an uncommitted probe; (4) add the new case to row S2 of the claim-to-test table and record the finding in this log. Nit 4 is done in the commit that adds this entry. Nits 1 to 3 are queued in `PROGRESS.md`, because each is a further code or test change outside the scope the owner set, and the pull request was merged at the head Opus 5.5 had reviewed.
- **Linux CI**, Ubuntu 24.04, Python 3.12.3, jax and jaxlib 0.11.2, numpy 2.5.3 and scipy 1.18.1 (the locked versions of the local runs). Run 38330860880 on `5a03241`: lint, format and 482 tests passed (pytest 109 s). Run 38333979012 on `8117f39`: 483 tests passed (149 s). Run 38334405621 on the merge commit on `main`: 483 tests passed (139 s). Every gate passed on Linux with no change to any value. The runner is noisy (109 s to 149 s for almost the same suite); the slowest new tests take seconds on the laptop, and no test came near the 60 s limit of D-036.
- **Merge.** Pull request #2 was marked ready and merged with a merge commit, `b66f6f9` (parents `0245058` and `8117f39`; no squash, D-033.3), at 2026-10-12 18:55 local time, on the owner's instruction. The branch `milestone-3-integrators` was not deleted, because the owner did not ask for it.
