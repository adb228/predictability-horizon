---
name: stage1-physics-ground-truth
overview: Build and validate the Stage 1 double-pendulum physics and Lyapunov-analysis foundation in JAX, with SciPy as an independent reference. Target 3–4 weeks at 15+ hours/week, but advance only when each evidence gate passes.
todos:
  - id: repo-baseline
    content: Create the reproducible JAX/uv repository baseline, project documentation, MIT license, tests, Ruff, and CI.
    status: pending
  - id: derive-dynamics
    content: Document and independently cross-check the mechanics, then support and review the user-authored double-pendulum equations.
    status: pending
  - id: integrate-validate
    content: Implement DOP853 and batched RK4 and pass all five physics validation gates.
    status: pending
  - id: lyapunov-methods
    content: Specify, support, and review the user-authored Benettin and variational/QR estimators, including known-system checks.
    status: pending
  - id: map-regimes
    content: Measure FTLE distributions, sensitivity, and energy-dependent regular/chaotic regimes without choosing thresholds post hoc.
    status: pending
  - id: publish-stage1
    content: Generate all Stage 1 results and figures reproducibly, write the report, complete interview checks, and prepare the approved release tag.
    status: pending
isProject: false
---

> **Note added when this file was saved (2026-10-11, at the owner's request).** Everything from the heading below is the plan exactly as it appeared in the Plans panel (`stage1-physics-ground-truth_a726b4e3.plan.md`): the plan approved on 2026-10-10 with the amendments D-008 to D-012, which the text already contains. Where this file and `docs/DECISIONS.md` differ, the decision log wins. Known differences:
>
> 1. **Authorship.** The plan says the owner writes the equations of motion and the Lyapunov estimators and the agent reviews. D-039 replaced that: the owner derives and verifies, the agent implements, and the agent walks the owner through the math before implementing (brief §7).
> 2. **Repository.** The plan says to create the public repository after milestone 4. It already exists as a private one (D-023) and becomes public after that gate (D-012).
> 3. **Status.** The `todos` in the front matter are as they were on the day the plan was created. Current status is in `PROGRESS.md`.
> 4. **Links.** Links in the body are relative to the repository root, not to this file.

# Stage 1 Physics Ground Truth

## Agreed setup
- Name the repository `predictability-horizon`, use the `horizon` Python package, MIT license, and GitHub Actions from day one.
- Use JAX for vectorized dynamics, RK4, Jacobians, and Lyapunov calculations; use SciPy DOP853 as the independent reference and SymPy only as a derivation cross-check.
- Defer a symplectic integrator; document why explicit Verlet is unsuitable and why implicit midpoint is outside Stage 1 scope.
- You will write the equations of motion and Lyapunov estimators after short conceptual walkthroughs; I will define interfaces/tests and review the implementations. Stage 2 metrics remain deferred.

## Milestones and evidence gates
1. **Reproducible repository baseline**
   - Save the authoritative brief to [`docs/BRIEF.md`](docs/BRIEF.md), initialize [`pyproject.toml`](pyproject.toml), [`PROGRESS.md`](PROGRESS.md), [`docs/DECISIONS.md`](docs/DECISIONS.md), MIT [`LICENSE`](LICENSE), package/test structure, deterministic configuration, and GitHub Actions.
   - Gate: a clean `uv` environment can import `horizon`, run Ruff, and pass a smoke test locally and in CI.

2. **Derived and cross-checked mechanics**
   - Document the Lagrangian, mass-matrix equations, energy, nondimensionalization, and small-angle modes in [`docs/derivations/double_pendulum.md`](docs/derivations/double_pendulum.md).
   - Before implementation, review the state convention, final equations, invariants, and tests; then you implement the equations in [`src/horizon/systems/double_pendulum.py`](src/horizon/systems/double_pendulum.py).
   - Gate: SymPy-generated residual checks agree at seeded random states, equilibria are stationary, and the linearization yields `ω² = 2 ± √2` for default parameters.

3. **Reference and fast integration**
   - Add the DOP853 wrapper and vectorized fixed-step JAX RK4 in [`src/horizon/integrators.py`](src/horizon/integrators.py), with explicit state shapes and time units.
   - Gate: RK4 shows fourth-order global convergence and meets a predeclared agreement tolerance against DOP853 over a stated horizon.

4. **Physics validation suite**
   - Add energy-drift, small-angle FFT, reversibility, convergence, and integrator-agreement tests under [`tests/`](tests/), with committed scripts recording measured values rather than handwritten results.
   - Gate: every validation passes without silently relaxing thresholds; long-horizon energy drift is measured and reported from the configured run. After this gate, create the public `predictability-horizon` repository.

5. **Two independent Lyapunov estimators**
   - Specify the wrap-safe difference metric, perturbation convention, transient handling, renormalization interval, and QR cadence in [`docs/derivations/lyapunov.md`](docs/derivations/lyapunov.md). Integrated angles remain unwrapped; wrapping is applied only when computing differences.
   - After a conceptual walkthrough, you implement Benettin and tangent-space/QR methods in [`src/horizon/lyapunov.py`](src/horizon/lyapunov.py). Run the pendulum variational/QR spectrum in canonical `(q, p)` coordinates; if a noncanonical `(θ, ω)` diagnostic is retained, explicitly demonstrate exponent-sum convergence with integration time.
   - Gate: validate on a linear system whose exponents equal the real parts of its eigenvalues, a harmonic oscillator with spectrum `(0, 0)`, and Lorenz with `λ₁ ≈ 0.906` and exact sum `−(σ + 1 + β)`. Then require both pendulum methods to agree on the largest exponent within a predeclared tolerance and document convergence of pairing and zero sum.

6. **Energy regimes and FTLE variability**
   - Run pilot scans before fixing energy bands or a regular/chaotic FTLE threshold. Perform one-at-a-time sensitivity sweeps around a declared default over timestep, perturbation magnitude, renormalization/QR cadence, transient length, window length, and phase-space norm, followed by one combined worst-case run rather than a full parameter grid.
   - Gate: selected bands and classification rules are justified in [`docs/DECISIONS.md`](docs/DECISIONS.md), regular trajectories are handled explicitly, and conclusions remain stable across reasonable numerical settings.

7. **Figures, report, and release**
   - Add one deterministic Stage 1 entry point under [`experiments/stage1/`](experiments/stage1/) that writes JSON results to [`results/`](results/) and regenerates the divergence, running-estimate, energy-regime, and flip-time figures in [`figures/`](figures/).
   - Write [`reports/stage1.md`](reports/stage1.md) from measured outputs, update [`README.md`](README.md), [`PROGRESS.md`](PROGRESS.md), and [`docs/DECISIONS.md`](docs/DECISIONS.md), then run the full reproducibility command and CI.
   - Gate: all Stage 1 definition-of-done items pass; you answer three interview questions; tag `stage1-v1.0` only after explicit approval.

## Main risks and early detection
- **Wrong equations, signs, or coordinate conventions:** independent SymPy residuals, equilibrium checks, small-angle modes, and energy conservation catch these before trajectory generation.
- **Numerical artifacts mistaken for chaos:** validate both estimators on known systems and require sensitivity sweeps plus estimator agreement before reporting any exponent.
- **Mixed phase space hidden by averages:** inspect per-initial-condition and windowed FTLE distributions before choosing bands or thresholds; never collapse regular and chaotic trajectories into one global `λ`.

## Scope control
Stages 2–4 remain untouched. No Stage 1 result, tolerance, energy band, classification threshold, or citation will be filled in before its generating script has run and its evidence has been reviewed.