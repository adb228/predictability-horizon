# Project Brief: Predictability Horizon

*Paste this as your first message to Cursor (Agent mode), then save it in the repo as `docs/BRIEF.md` so every future session can be pointed at it with `@docs/BRIEF.md`.*

---

## 0. Read this first: how we work together

You are my research engineering partner on a four-stage project. Before writing any code, do the **calibration protocol** in Section 9. Do not skip it, and do not start Stage 1 until I approve your plan.

The single most important rule: **I need to be able to explain and defend every line of this repo in a research interview, unaided.** That means:

- Every non-obvious choice comes with a short **why** (2–3 lines, not an essay). Keep explanations brief; long-winded explanations make things harder to follow, not easier.
- When there are two reasonable options, give me the tradeoff in a few lines and a recommendation. Don't silently pick.
- Flag anything you're unsure about. Say "I'm not certain" rather than guessing confidently. This applies especially to physics equations and citations.
- Never fabricate, estimate, or "illustrate" a result. If something hasn't run, say it hasn't run. Every number and figure in this repo must come from a committed script.

---

## 1. Who I am

Owner: physics and math undergraduate; derives and verifies the mathematics.

---

## 2. The research question

> **How far ahead can a learned model predict a chaotic system, and is that limit set by the physics or by the model?**

Chaotic systems have a hard predictability limit: tiny errors grow like `e^(λt)`, where `λ` is the largest Lyapunov exponent. The natural unit of time is the **Lyapunov time** `1/λ`. Most learned world models are evaluated on raw error, which conflates two things:

1. **Error the physics forces on you** (intrinsic chaos amplifying any small initial error), and
2. **Error the model adds** (bad dynamics, wrong structure, compounding approximation error).

This project measures learned models **in units of Lyapunov time** and separates those two sources of error. It climbs from pure physics to a learned, action-conditioned world model used for planning.

**Why this framing:** it turns a toy system into a finding. "My model predicts the double pendulum" is a tutorial. "My model's horizon is X Lyapunov times, within Y% of the chaos-imposed limit, and here's what closes the gap" is research.

---

## 3. The four-stage ladder

Each stage must be **a complete, finished result on its own**: a clean figure, a short writeup in `reports/`, and a git tag (`stage1-v1.0`, etc.). Later stages reuse earlier code. Nothing is throwaway.

### Stage 1: The physics ground truth (no ML)

**Goal:** a trustworthy double-pendulum simulator and a rigorous Lyapunov exponent estimate. Everything later is measured against this, so it has to be right.

**System:**
- State: `(θ1, θ2, ω1, ω2)`. Parameters: `m1, m2, L1, L2, g`.
- Default to nondimensional units: `m1 = m2 = 1`, `L1 = L2 = 1`, `g = 1` (time in units of `√(L/g)`). Explain if you recommend otherwise.
- Energy (for validation, not as a substitute for deriving the equations):
  - `T = ½(m1+m2)L1²ω1² + ½m2L2²ω2² + m2L1L2ω1ω2·cos(θ1−θ2)`
  - `V = −(m1+m2)gL1·cos θ1 − m2gL2·cos θ2`
- **Derive the equations of motion from the Lagrangian** and show me the derivation in `docs/derivations/double_pendulum.md`. Do not paste equations from memory without tests that would catch a sign error. Optionally use `sympy` to derive and cross-check.

**Integrators:**
- **Reference:** `scipy.integrate.solve_ivp` with `DOP853`, `rtol=1e-12`, `atol=1e-12`. This is the ground-truth generator.
- **Fast batched:** fixed-step RK4 in the main framework (vectorized over many initial conditions).
- **Optional, for discussion:** a symplectic integrator. Note that the double pendulum Hamiltonian is **non-separable** (kinetic energy depends on `θ1−θ2`), so explicit leapfrog/Störmer–Verlet doesn't apply directly. Implicit midpoint or Gauss–Legendre would. Explain the tradeoff before building it; it may not be worth it.

**Validation tests (gate to moving on; all must pass in `pytest`):**
1. **Energy conservation:** relative energy drift under the reference integrator stays below ~`1e-9` over a long horizon (e.g. `t = 1000`). Report the actual number.
2. **Small-angle limit:** for `m1 = m2`, `L1 = L2`, linearized normal-mode frequencies are `ω² = (g/L)(2 ± √2)`. Simulate small oscillations, take an FFT, and check that the peaks match.
3. **Time reversibility:** integrate forward, flip velocities, integrate back, recover the initial state (in a non-chaotic regime, or over a short horizon in a chaotic one).
4. **Convergence order:** RK4 global error should scale as `dt⁴` against the reference. Show the log-log slope.
5. **Agreement:** RK4 at the chosen `dt` matches DOP853 to a stated tolerance over a stated horizon.

**Lyapunov analysis:**
- **Method A, two-trajectory (Benettin):** perturb by `d0 ~ 1e-9`, renormalize every `τ`, average `log(d/d0)/τ`. Simple and intuitive.
- **Method B, variational / tangent-space with QR:** integrate the linearized equations using the Jacobian (via autodiff) and re-orthonormalize with QR to get the **full spectrum**.
- **Built-in sanity check:** for an autonomous 2-degree-of-freedom Hamiltonian system, the spectrum should look like `(λ, 0, 0, −λ)`: exponents come in ± pairs and sum to ~0. Test this.
- Methods A and B should agree on the largest exponent. If they don't, that's a bug, not a finding.
- **λ is regime-dependent, so never report one global number.** Even at fixed energy, the double pendulum's phase space is mixed: regular (quasi-periodic) islands coexist with a chaotic sea. So:
  - Report `λ` **per energy band**, across an ensemble of initial conditions.
  - Show the **distribution of finite-time Lyapunov exponents (FTLEs)** over windows of a chosen length, not just the long-time mean, to capture variability along a trajectory.
  - Classify initial conditions as regular vs chaotic (e.g. by FTLE threshold) and state how regular ones are handled in Stage 2 datasets.
  - Stage 2 normalizes by the `λ` of the **specific band** each dataset comes from, and reports how sensitive conclusions are to that choice.

**Stage 1 figures:**
1. **Divergence plot:** log separation of two nearby trajectories vs time, with a fitted slope `λ` overlaid. This is the hero figure.
2. Running Lyapunov estimate converging over time, with both methods.
3. `λ` vs total energy: near-zero at low energy (quasi-periodic), positive at high energy (chaotic). Show the transition.
4. **Flip-time map:** a grid of initial `(θ1, θ2)` with zero velocity, colored by time until either pendulum flips over. It's a classic fractal image and a great README visual.

**Stage 1 deliverable:** `reports/stage1.md` with the four figures, validation results, the measured `λ` (with uncertainty, from multiple initial conditions in the same energy band), and the Lyapunov time used by all later stages.

---

### Stage 2: Learned dynamics, measured in Lyapunov times

**Goal:** train models on simulated trajectories and measure how long their predictions stay valid, **in Lyapunov times**, compared against the chaos-imposed limit.

**Data:**
- Generate trajectories from the Stage 1 reference integrator in a **chaotic energy band** (chosen from the `λ`-vs-energy plot).
- **Split by initial condition, never by time within a trajectory.** Time-based splits leak information because nearby windows of one trajectory are highly correlated.
- Include a **held-out energy band** for a generalization test.
- **Angle representation:** handle wrapping. Use `(sin θ, cos θ)` features or wrap carefully. Compute all errors in a wrap-safe space. This is a common silent bug; test for it.

**Models (start with the first two):**
1. **MLP vector field:** learns `ds/dt = f(s)`, integrated with RK4 (neural-ODE style). Alternative: direct next-state or delta prediction. Recommend one and justify it.
2. **Hamiltonian Neural Network** (Greydanus et al., 2019): learns a scalar `H(q, p)` and gets dynamics through the symplectic gradient. Structure-preserving, so it should conserve its learned energy.
   - **It must use canonical coordinates.** `(θ, ω)` is not canonical for the coupled pendulum: the momenta are `p_i = ∂L/∂ω_i`, which mix both angular velocities. Convert the data to `(θ1, θ2, p1, p2)` and add a test that the conversion round-trips.
   - Angle embeddings are allowed **inside** the network (feed `sin q, cos q` as input features to `H`), but `q̇ = ∂H/∂p` and `ṗ = −∂H/∂q` must be computed by autodiff with respect to the raw `q` through that embedding. Never treat `(sin θ, cos θ)` as the coordinates themselves.
   - For a fair comparison, give the MLP the same `(q, p)` inputs as one variant, so the HNN's advantage (or lack of one) comes from structure rather than coordinates.
3. *Optional:* Lagrangian Neural Network (Cranmer et al., 2020).
4. **Baselines:** persistence (predict no change) and the true system linearized at the start point. These keep the "learned" results honest.

**Training ablation (key experiment):** one-step loss vs multi-step rollout loss (e.g. unroll `k = 1, 5, 20`). Hypothesis: rollout training extends the horizon. Test it; don't assume it.

**Metrics:**
- **Valid Prediction Time (VPT):** time until normalized error exceeds a threshold `ε` (e.g. `ε = 0.3`), reported **in Lyapunov times** (`t · λ`). Report sensitivity across several `ε` values; one threshold isn't enough.
- **Matched-perturbation physical baseline (not a hard bound):** take the true system, start it from an initial condition perturbed by an error of the model's size, and measure when *that* crosses `ε`. Theory says `t*·λ ≈ ln(ε / e0)`. **This is a reference curve, not an upper bound.** A scalar error size ignores perturbation direction, temporal correlation, and systematic model bias, so a model can in principle beat it (e.g. if its errors lie along contracting directions). Compute it two ways: (a) perturbing along the model's **actual one-step error vector**, and (b) averaging over many **random directions** of the same magnitude. Report both, and never call it an "oracle" or "limit" in writeups.
- **The decomposition:** compare the model's VPT to the oracle horizon. If the model's error grows at rate `λ`, the model is **chaos-limited** (good). If it grows faster, it's **model-limited**. Plot the model's log-error vs time against the oracle's on the same axes.
- **Learned model's own Lyapunov exponent:** compute `λ` of the learned vector field using the Stage 1 code (autodiff the learned `f`). Does the model reproduce the right amount of chaos, even after its trajectories diverge?
- **Long-run statistics:** after trajectories diverge, does the model still stay on the right energy surface and visit the right regions of state space? (Energy drift and state-distribution histograms vs ground truth.) Short-term accuracy and long-term statistical fidelity are different properties, and world models need both.

**Rigor:**
- At least **5 seeds** per configuration. Report means with confidence intervals.
- **Matched parameter counts and data budgets** across models, so comparisons are fair.
- Log every run, including failures. Don't tune hyperparameters on the test set.

**Stage 2 hero figure:** VPT in Lyapunov times for each model, next to the oracle horizon, with a log-error-growth panel showing chaos-limited vs model-limited behavior.

---

> **Stages 3 and 4 are provisional sketches.** Do not design them in detail, write code for them, or make architecture decisions until Stage 2 has produced a stable, trusted metric. Their purpose here is only to show where the ladder goes. We will rewrite these sections with what Stage 2 teaches us.

### Stage 3: From pixels (a real world model)

**Goal:** learn the same dynamics from rendered images instead of the true state, and measure how much predictability horizon is lost.

**Setup:**
- Render frames (start with 64×64 grayscale: two rods and two bobs on a plain background).
- A single frame contains no velocity information. Stack 2–3 frames or use a recurrent encoder, and explain the choice.
- Known ambiguity: when the rods overlap, the image doesn't uniquely determine the state. Note it and measure whether it matters.

**Architecture (recommend a starting point and justify it):**
- **Option A:** autoencoder (or VAE) + latent dynamics model, trained with multi-step latent rollouts. A simplified Dreamer/World Models setup. Probably the right start.
- **Option B:** JEPA-style latent prediction with no pixel reconstruction. Needs collapse prevention (EMA target encoder or VICReg-style regularization). More modern, more fragile.

**Evaluation (pixel MSE is misleading here):** in chaotic systems, a model that predicts a blurry average can score well on MSE while being useless.
- **Linear probe:** fit a linear map from latent to the true `(sin θ, cos θ, ω)`. Does the latent recover the physics?
- **VPT in Lyapunov times** using probed state, compared directly to the Stage 2 state-based models.
- **Lyapunov exponent of the latent dynamics:** does the latent space have the right chaos?

**Stage 3 hero figure:** horizon comparison across true state → learned state model → pixel world model, all in Lyapunov times, with sample rollout filmstrips (true vs predicted frames).

---

### Stage 4: Action-conditioned model + planning

**Goal:** turn the world model into an **action model** and plan with it. Test whether the useful planning horizon is bounded by the predictability horizon from earlier stages.

**System: the acrobot**, which is the same double pendulum with torque applied at the elbow joint only (underactuated). Build it by extending the Stage 1 simulator rather than importing Gym's discrete-action version, so the physics, validation, and Lyapunov tooling all carry over.

**Task:** swing-up from hanging down to upright, then balance.

**Approach:**
- Learn `s_{t+1} = f(s_t, a_t)` from state first. An **ensemble** of models gives uncertainty estimates (PETS-style, Chua et al., 2018).
- Plan with **MPC**: random shooting first as a baseline, then the cross-entropy method (CEM).
- *Optional extension:* plan from pixels using the Stage 3 world model (PlaNet-style).

**Caution, the Lyapunov time does not carry over automatically.** Controlled dynamics depend on the state, the action sequence, and the feedback policy. Near the upright balance point under good control, the closed-loop system can even be locally contracting. Measure FTLEs **along the actual controlled trajectories** (open-loop under the planned actions, and closed-loop under the MPC policy) rather than reusing the passive pendulum's `λ`, and state which one each plot uses.

**Key experiment:** sweep the MPC planning horizon `H` and plot task success vs `H` expressed in Lyapunov times. Hypothesis: success peaks near the model's predictability horizon and degrades beyond it, as the planner starts trusting rollouts that have already diverged. This ties the whole ladder together.

**Stage 4 hero figure:** success rate vs planning horizon (in Lyapunov times), with the Stage 2/3 predictability horizon marked as a vertical line, plus a GIF of a successful swing-up.

---

## 4. Tech stack (confirm with me during calibration)

- **Python 3.12+** (D-027), environment managed with **`uv`**.
- **Framework, my default preference: JAX** (`jax`, `equinox` for models, `optax` for optimizers, optionally `diffrax` for ODE solving). Why JAX: `jax.jacfwd` gives exact Jacobians for the variational Lyapunov method, and `vmap` batches thousands of initial conditions cheaply. If you think PyTorch (+ `torchdiffeq`) is better for this project, make the case briefly.
- `scipy` for the reference integrator, `sympy` to derive and cross-check equations of motion.
- `pytest` + `hypothesis` for tests, `ruff` for lint and format.
- `matplotlib` for figures (consistent style file, colorblind-safe palette, vector output for reports).
- Configs: simple dataclasses or YAML. Avoid heavy config frameworks unless they earn their place.
- Experiment tracking: local JSON results to start. `wandb` optional later. Ask first.
- **Compute:** Stages 1–2 should run on a laptop CPU. I have GPU access for Stages 3–4.
- **Ask before adding any dependency** not listed here.

---

## 5. Repo structure (proposed; adjust if you have a reason)

```
predictability-horizon/
├── pyproject.toml
├── README.md                  # hero figure, one-paragraph finding, reproduce command
├── PROGRESS.md                # running log: what's done, what's next, open questions
├── docs/
│   ├── BRIEF.md               # this file
│   ├── DECISIONS.md           # short decision log: choice, alternatives, why
│   └── derivations/           # math writeups (EOM, Lyapunov methods, metrics)
├── src/horizon/
│   ├── systems/               # double_pendulum.py, acrobot.py
│   ├── integrators.py
│   ├── lyapunov.py
│   ├── data.py                # deterministic, seeded dataset generation
│   ├── metrics.py             # VPT, oracle horizon, wrap-safe errors
│   ├── render.py              # Stage 3 frame rendering
│   ├── models/                # mlp.py, hnn.py, latent.py, ensemble.py
│   └── plan/                  # mpc.py (random shooting, CEM)
├── experiments/
│   ├── stage1/                # one script per figure + make_figures entry point
│   ├── stage2/
│   ├── stage3/
│   └── stage4/
├── results/                   # JSON outputs (small); large artifacts gitignored
├── figures/
├── reports/                   # stage1.md ... stage4.md
└── tests/
```

**Why one repo with tagged stages instead of four repos:** every stage reuses the simulator, Lyapunov code, and metrics. A single repo with a visible progression reads as a sustained research effort rather than four disconnected weekend projects.

---

## 6. Engineering rules (non-negotiable)

1. **Reproducibility:** every figure and every number in a README or report is produced by a committed script with a committed config and an explicit seed. There should be one command per stage that regenerates all its figures.
2. **No fabricated results, ever.** No placeholder numbers, no "expected" plots, no illustrative values. If it didn't run, the report says so.
3. **Tests before trust:** physics validation tests must pass before any ML is trained on that data.
4. **Data is generated, not committed:** datasets are rebuilt deterministically from seeds. Gitignore anything large.
5. **Small, meaningful commits** with clear messages. One logical change per commit. This repo's history is part of what reviewers see, so it should reflect real, incremental work.
6. **Negative results are results.** If rollout training doesn't help, or the HNN loses to the MLP, we report it honestly with the evidence.
7. **No silent tuning:** don't change hyperparameters to make a plot look better without logging it in `DECISIONS.md`.
8. **Type hints and docstrings** on public functions. Docstrings state units and array shapes.
9. **Citations:** before any paper is cited in a README or report, flag it for me to verify (title, authors, year). Don't trust your own memory of citations.

---

## 7. Teaching mode

- For the **core numerical pieces** (equations of motion, integrators, Lyapunov estimators, VPT and oracle metrics), walk me through the math *before* you write the code, briefly. I may want to write some of these myself while you review. Ask which ones.
- At the end of each milestone, give me **3 interview-style questions** I should be able to answer about what we just built (e.g. "Why do the Lyapunov exponents come in ± pairs?", "Why split by initial condition instead of by time?"). Don't give the answers unless I ask.
- If I push back on a choice, defend it if you think you're right. Don't just agree. I'd rather be corrected than flattered.

---

## 8. Session protocol (every session after the first)

1. Read `docs/BRIEF.md`, `PROGRESS.md`, and `docs/DECISIONS.md`.
2. In ≤5 lines, state where we are and what you propose for this session.
3. Wait for my go-ahead on anything bigger than a small fix.
4. At the end, update `PROGRESS.md` (done / next / open questions) and log any decisions in `DECISIONS.md`.

---

## 9. Calibration protocol (do this now, before any code)

1. **Restate** the project goal and the Stage 1 scope in your own words, in ≤10 lines, so I can check we're aligned.
2. **Ask me up to 8 clarifying questions.** Cover at least:
   - JAX vs PyTorch
   - my weekly time budget and target dates per stage
   - which core pieces I want to write myself vs review
   - how deep the math explanations should go
   - repo name and license
   - whether to set up CI (GitHub Actions running tests and lint) from day one
3. **Propose a Stage 1 plan** as a short milestone list (each milestone = something testable), with the 2–3 biggest technical risks and how you'd catch them early.
4. **Stop and wait** for my approval.

---

## 10. Prior work to be aware of (verify before citing; we're not claiming novelty for these)

- Benettin et al. (1980) and Wolf et al. (1985): Lyapunov exponent estimation
- Chen et al. (2018): Neural ODEs
- Greydanus et al. (2019): Hamiltonian Neural Networks
- Cranmer et al. (2020): Lagrangian Neural Networks
- Pathak et al. (2018): model-free prediction of chaotic systems (reservoir computing), reported in Lyapunov times
- Vlachas et al. (2020): RNNs vs reservoir computing for chaotic forecasting
- Ha & Schmidhuber (2018): World Models
- Hafner et al.: PlaNet (2019) and the Dreamer line
- Chua et al. (2018): PETS (ensembles + MPC)

**Where this project tries to add something:** consistently measuring learned and pixel-based world models in Lyapunov time, explicitly separating chaos-limited from model-limited error, and connecting predictability horizon to the useful planning horizon in MPC. If you know of prior work that already does exactly this, tell me. I'd rather know now.

**These novelty claims are provisional.** Before the Stage 2 report makes any claim of contribution, there is a dedicated **literature review milestone**: verify every paper above, then search adjacent work (Lyapunov-normalized evaluation of learned models, Lyapunov spectra of learned dynamics, predictability horizons in model-based RL and MPC). Record the findings in `docs/related_work.md` and narrow our claims to match.

---

## 11. Definition of done, per stage

A stage is done when:
- [ ] All validation tests pass in CI (or locally if no CI)
- [ ] One command regenerates every figure for the stage
- [ ] `reports/stageN.md` contains the hero figure, a one-paragraph finding, methods, and limitations
- [ ] The README is updated: hero figure at the top, finding in plain words, reproduce command
- [ ] `PROGRESS.md` and `DECISIONS.md` are current
- [ ] The commit is tagged `stageN-v1.0`
- [ ] I can answer your three interview questions for that stage without help
