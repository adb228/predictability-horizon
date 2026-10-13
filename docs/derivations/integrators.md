# Integrators: RK4, the DOP853 reference, and how far two integrators can agree

This note states every claim that a milestone 3 test checks. Section 8 maps each claim to the test that guards it. The decisions behind the tests are in `docs/DECISIONS.md` (D-042 to D-049). **The values in section 7 were declared in that log before any test ran against the implementation; they are not measurements.** The calibration that informed them is uncommitted lab work, described and labelled as such in the log, and no calibrated number appears here. If this note and `src/horizon/tolerances.py` disagree about a value, the register wins.

## 1. Conventions and units

The system is `ẏ = f(y, p)`: autonomous, `y` an array of shape `(n,)` and dtype `float64`, `p` the parameters (any JAX pytree: a `Params` for the pendulum, `None` for the exact problems of section 3), and `f` returning an array of the same shape as `y`. One state is integrated per call (D-028). Batching is done by the caller with `jax.vmap`, over states and, with a `Params` whose fields are arrays, over parameter sets.

**Time is in whatever unit `f` uses.** For the pendulum that is `√(L1/g)` (section 8 of `docs/derivations/double_pendulum.md`), which is 1 for the default parameters. The gates that use non-default parameters therefore state `dt` and `T` in units of `t_s = √(L1/g)` and measure errors on the nondimensional state `(θ1, θ2, ω1 t_s, ω2 t_s)`.

`horizon.integrators` has three functions.

- `rk4_step(f, state, params, dt)` is one classical RK4 step.
- `rk4_trajectory(f, state0, params, dt, n_out, stride)` returns an array of shape `(n_out + 1, n)`. Row `k` is the state at `t = k · stride · dt`, computed as an integer multiple and never accumulated; row 0 is `state0`. `f`, `n_out` and `stride` are static, so a new function object or a new count recompiles. `dt` may be negative.
- `solve_reference(f, state0, params, t_eval, *, rtol=1e-12, atol=1e-12)` runs SciPy's DOP853 over `(0, t_eval[-1])` and returns an array of shape `(len(t_eval), n)`. It requires `t_eval[0] ≥ 0` and `t_eval` strictly increasing, raises `RuntimeError` if SciPy reports failure, and turns any warning raised during the solve into an exception, so that SciPy's silent clamp of `rtol` (section 4) cannot change a tolerance unnoticed.

## 2. Classical RK4

### 2.1 The scheme

$$k_1 = f(y_n),\quad k_2 = f\bigl(y_n + \tfrac h2 k_1\bigr),\quad k_3 = f\bigl(y_n + \tfrac h2 k_2\bigr),\quad k_4 = f\bigl(y_n + h k_3\bigr),\qquad y_{n+1} = y_n + \tfrac h6\,(k_1 + 2k_2 + 2k_3 + k_4).$$

### 2.2 Linear problems: the stability polynomial

For `f(y) = A y` the stages are `k1 = A y`, `k2 = (A + (h/2)A²) y`, `k3 = (A + (h/2)A² + (h²/4)A³) y` and `k4 = (A + hA² + (h²/2)A³ + (h³/4)A⁴) y`. Adding them with the weights `1, 2, 2, 1`, the coefficients of `A, A², A³, A⁴` are `6`, `3h`, `h²` and `h³/4`. Multiplying by `h/6` turns them into `h`, `h²/2`, `h³/6` and `h⁴/24`, so

$$y_{n+1} = R(hA)\,y_n,\qquad R(z) = 1 + z + \tfrac{z^2}{2} + \tfrac{z^3}{6} + \tfrac{z^4}{24},$$

the Taylor polynomial of `eᶻ` through `z⁴`. The one-step error on a linear constant-coefficient problem is `e^{hA} − R(hA) = O(h⁵)`. This is claim R1, and it is an exact identity, so it is checked to the `IDENTITY` tolerance.

### 2.3 Order, local error and global error

A one-step method has order `p` if its step differs from the exact flow by `O(h^{p+1})`. The four stages and the weights of section 2.1 satisfy the eight order conditions of orders 1 to 4 (one, one, two and four conditions), so the Taylor series of the step agrees with that of the exact flow through `h⁴` and the **local error is `O(h⁵)`**. Reaching a fixed time `T` takes `T/h` steps, and the standard convergence theorem for one-step methods (Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*, section II.3) gives

$$\|y(T) - y_N\| \;\le\; \frac{C\,h^4}{L}\,\bigl(e^{LT} - 1\bigr),$$

where `L` is a Lipschitz constant of `f` along the solution and `C` depends on derivatives of `f`. Three consequences are used:

- Halving `h` divides the error by 16, so the slope of `log(error)` against `log(h)` tends to 4. This is the convergence claim (G1, G2).
- The factor `e^{LT}` is why long horizons in chaotic motion are expensive (section 5).
- The slope is *asymptotic*. For larger `h` the higher-order terms of the error can dominate and the measured slope can sit below 4 (pre-asymptotic behaviour). That is why the grids, amplitudes and slope bands are declared in advance (D-045) and not tuned afterwards.

### 2.4 Why linear problems are not enough

Take the scheme with every stage fraction equal to `½` and weights `(0, ⅓, ⅓, ⅓)`: `k2 = f(y + (h/2)k1)`, `k3 = f(y + (h/2)k2)`, `k4 = f(y + (h/2)k3)`, `y_{n+1} = y_n + h(k2 + k3 + k4)/3`. On `f(y) = Ay` its stages are `k2 = (A + (h/2)A²)y`, `k3 = (A + (h/2)A² + (h²/4)A³)y` and `k4 = (A + (h/2)A² + (h²/4)A³ + (h³/8)A⁴)y`, and the update gives the **same polynomial `R`** as in section 2.2. So it is fourth order on every linear constant-coefficient problem. But `Σ bᵢcᵢ² = (1/3)(1/4)·3 = 1/4`, not the `1/3` that the third-order condition requires, so on nonlinear problems it is only second order. Any gate made of linear problems alone passes it.

That is why the exact-solution gate has two nonlinear problems, and test N1 runs this wrong scheme (called W2 there) to show that it passes the two linear problems and fails the two nonlinear ones. N1 also runs two schemes with a mistyped stage, which are not fourth order on anything: W1 evaluates `k3` at `y + (h/2)k1` instead of `y + (h/2)k2`, and W3 evaluates `k4` at `y + (h/2)k3` instead of `y + h k3`. N1 asserts only that the exact-solution gate fails for them. The logistic problem is autonomous and scalar; for such problems the two order-4 trees `[[τ], τ]` and `[[τ, τ]]` have the same elementary differential, `f'' f' f²`, so it checks only the combination of their two conditions. The cubic rotation is a genuine two-dimensional system, which is why it is there too (D-044).

## 3. Problems with exact solutions

Each solution is written in closed form in `tests/integrator_checks.py`, and tests X1 to X4 check with `jax.jvp` that it satisfies its differential equation and its initial condition. They use no integrator at all, so a typo in a formula cannot hide inside the gate that uses it.

| Problem | Equation | `y(0)` | Solution |
|---|---|---|---|
| harmonic oscillator | `ẋ = v`, `v̇ = −4x` | `(1, 0.5)` | `x = x₀ cos 2t + (v₀/2) sin 2t`, `v = −2x₀ sin 2t + v₀ cos 2t` |
| linear, non-normal | `ẏ = A y`, `A = [[−1, 5], [0, −2]]` | `(1, −1)` | `y₂ = y₂(0) e^{−2t}`, `y₁ = y₁(0) e^{−t} + 5 y₂(0)(e^{−t} − e^{−2t})` |
| logistic | `ẏ = y(1 − y)` | `0.1` | `y = y₀ eᵗ / (1 − y₀ + y₀ eᵗ) = 1/(1 + 9e^{−t})` |
| cubic rotation | `ẋ = −(x² + y²) y`, `ẏ = (x² + y²) x` | `(1, 0.5)` | radius constant; angle `φ₀ + r₀² t`, with `r₀² = x₀² + y₀²` |

For the linear problem, `ẏ₂ = −2y₂` gives `y₂`, and `ẏ₁ = −y₁ + 5y₂(0)e^{−2t}` is solved with the integrating factor `eᵗ`. For the rotation, `d(x² + y²)/dt = 2xẋ + 2yẏ = 0`, and the polar angle obeys `φ̇ = (xẏ − yẋ)/(x² + y²) = x² + y²`, constant. The matrix `A` is non-normal (`A Aᵀ ≠ Aᵀ A`), so a method's linear stability cannot be read from eigenvalues alone.

**Claim.** On the grid `dt ∈ {0.04, 0.02, 0.01, 0.005}`, with outputs at the integer times `t = 0, 1, …, T`, RK4's error against each exact solution has a log-log slope in the declared band and, at the finest step, stays below the declared ceiling; and the DOP853 wrapper at its default tolerances is within the declared distance of each solution. The statistic is the maximum over output times and components of `|RK4 − exact|` (G1).

## 4. The DOP853 reference

SciPy's `DOP853` is an explicit Runge-Kutta method of order 8 with **12 stages** (SciPy 1.18.1). From the same stages it forms two embedded lower-order solutions, of orders 5 and 3. Their differences from the order-8 step, `err5` and `err3`, are scaled per component by

$$s_i = \mathrm{atol} + \max\bigl(|y_{n,i}|, |y_{n+1,i}|\bigr)\,\mathrm{rtol},$$

and combined into one number

$$e = \frac{|h|\,\|err_5\|^2}{\sqrt{\bigl(\|err_5\|^2 + 0.01\,\|err_3\|^2\bigr)\,n}}.$$

A step is accepted if `e < 1`. The next step size is `h · min(10, 0.9 e^{−1/8})` (capped at a factor 1 right after a rejection), because the error estimate scales like `h⁸`; a rejected step is retried with `h · max(0.2, 0.9 e^{−1/8})`. So steps shrink where the motion is sensitive and grow where it is smooth.

Three points matter for the tests.

- `rtol` and `atol` bound the **local** error per step, not the global error. In chaotic motion later steps amplify earlier local errors (section 5), so the reference is checked against a tighter run of itself (the validity rule of G2) and against exact solutions (G1).
- SciPy raises any `rtol` below `100·eps ≈ 2.22e-14` to that value and emits a `UserWarning`; `atol` is not clamped. The wrapper turns the warning into an error (S4), so a request for `1e-14` fails instead of running at `2.22e-14`.
- The reference is **pointwise**: it is accurate at the requested times as long as errors have not been amplified past the tolerance. Beyond some horizon in a chaotic state, compare statistics, not trajectories (known limitation, carried into milestones 5 and 6 and into the Stage 2 valid-prediction-time measurements).

## 5. Chaos limits how far two integrators can agree

Nearby trajectories of a chaotic system separate like `δ(t) ≈ δ₀ e^{λt}`, with `λ` the (finite-time) growth rate along the trajectory. An integrator whose per-step error is `ε` injects a perturbation of size `ε` every step of length `h`; the dynamics amplify each one, so its deviation from the true solution is about

$$e(t) \;\approx\; \sum_k \varepsilon\,e^{\lambda(t - t_k)} \;\approx\; \frac{\varepsilon}{\lambda h}\,\bigl(e^{\lambda t} - 1\bigr).$$

Two integrators A and B therefore agree to within `τ` only until

$$T^\ast \;\approx\; \frac1\lambda\,\ln\!\frac{\tau}{\varepsilon_{\rm eff}},\qquad \varepsilon_{\rm eff} = \frac{\varepsilon_A + \varepsilon_B}{\lambda h}.$$

- The dependence on accuracy is **logarithmic**: an integrator ten times more accurate buys only `ln 10 / λ ≈ 2.3` Lyapunov times (`1/λ` is the Lyapunov time).
- After `T*` the difference saturates at the size of the attractor, order one, and the two trajectories are simply different. That is a property of chaos, not a bug.
- So a tolerance and a horizon only make sense as a pair. The gate declares both (section 7), and the horizon is stated in units of `√(L1/g)` now and will be restated in Lyapunov times after milestone 5 measures `λ`; that is a restatement, not a change.

The agreement gate has two parts (G3, G4). (a) **Chaotic-ish states**: 30 seeded states released at rest with `E > 0` (above the pivot-height zero of the potential energy), with the statistic being the worst case over states, output times and components, because the maximum of a heavy-tailed quantity is deterministic and conservative. "Chaotic-ish" is not a classification; that is milestone 6. (b) **Regular states**: the 12 small-amplitude states of the convergence study, whose motion is close to a pair of linear oscillators, over a longer horizon.

## 6. Non-default parameters: the wiring check and the energy anchor

With all five parameters equal to one, a dropped, swapped or unused `params` changes nothing (D-028). G5 and G6 therefore use 10 seeded non-default parameter sets, 3 chaotic-ish states each, with `dt` and `T` in units of `t_s = √(L1/g)` of the set, and compare the nondimensional state.

RK4 and the wrapper are compared **with each other**, so an error that both share, for example the same dropped `params`, is invisible to that comparison. The anchor does not rest on the two agreeing. The true dynamics conserve the total energy (section 4 of the double-pendulum note), so for a correct integrator

$$\max_t \frac{|E(t) - E(0)|}{E_s} \;\le\; \text{declared tolerance},\qquad E_s = g\bigl((m_1 + m_2)L_1 + m_2L_2\bigr),$$

with `E` evaluated by `double_pendulum.energy` under that set's own parameters. A dropped or swapped parameter in either integrator, or in both, makes the integrator follow different dynamics, along which the true energy is not conserved. `E_s` is the largest possible `|V|`, a fixed scale, so the ratio is never inflated by a state whose own energy happens to be near zero.

## 7. Declared values (declared, not measured)

All values were logged in D-044 to D-047 before any test ran against the implementation, and they live in `src/horizon/tolerances.py`, pinned by a tripwire test (K1). Changing any of them needs Opus 5.5's approval first (D-025), and the owner has asked that none be changed after seeing results from the implementation.

| Gate | Declared value | Decision |
|---|---|---|
| exact problems: grid | `dt ∈ {0.04, 0.02, 0.01, 0.005}`, outputs at integer times | D-044 |
| exact problems: slope band | `[3.9, 4.1]` for each problem | D-044 |
| exact problems: horizons and ceilings on the error at `dt = 0.005` | harmonic oscillator `T = 10`, `3e-8`; linear `T = 5`, `2e-9`; logistic `T = 5`, `7.5e-12`; cubic rotation `T = 5`, `1.1e-8` | D-044 |
| DOP853 wrapper (default tolerances) against exact | `≤ 1e-10` on all four | D-044 |
| convergence: states | 12 states, `ω = 0`, `θ1, θ2` uniform in `[−0.4, 0.4]`, stream `[CASES_SEED, 2000]` | D-045 |
| convergence: setup | default parameters, `T = 10`, outputs at integer times, same `dt` grid as above | D-045 |
| convergence: reference | `rtol = atol = 1e-13`; self-check reference `2.5e-14` | D-045 |
| convergence: slope band | `[3.8, 4.2]` for every state | D-045 |
| convergence: validity rule | error at `dt = 0.005` at least `30×` the reference self-error, else the test fails as inconclusive | D-045 |
| agreement, chaotic-ish | `dt = 0.01`, `T = 8`, 30 states (stream `[CASES_SEED, 2001]`), reference `1e-12`, worst case `≤ 1e-4` | D-046 |
| agreement, regular | the 12 states above, `dt = 0.01`, `T = 50`, reference `1e-13`, worst case `≤ 1e-6` | D-046 |
| wiring: cases | 10 parameter sets (stream `[CASES_SEED, 2002]`), 3 states each (stream `[CASES_SEED, 2003]`) | D-047 |
| wiring: setup | `dt = 0.005 t_s`, `T = 2 t_s`, outputs at `0, t_s, 2t_s`, reference `1e-12` | D-047 |
| wiring: agreement and energy anchor | each `≤ 1e-5` | D-047 |

The `IDENTITY` tolerance class of `tolerances.py` is used for the exact identities (R1 to R5, S1, X1 to X4); it is not a new threshold. Gate comparisons (a statistic at most a ceiling, a slope inside a band, an error at least a floor) go through three small helpers, `check_at_most`, `check_in_band` and `check_at_least`, which fail on NaN and infinity and report the margin used, so that no test can quietly use a looser rule. K2 tests the helpers themselves.

## 8. Claims and the tests that check them

Tests are in `tests/test_integrators.py` (R, S), `tests/test_integrator_gates.py` (X, G, N) and `tests/test_integrator_register.py` (C, K); the shared definitions and measurements are in `tests/integrator_checks.py`, which the script `experiments/stage1/integrator_validation.py` also uses, so a gate and the number printed for it have one implementation.

| Claim | Section | Test | Rule |
|---|---|---|---|
| One RK4 step on a linear problem equals `R(hA) y` | 2.2 | R1 | `IDENTITY` |
| One RK4 step on a nonlinear polynomial field equals the same scheme in exact rational arithmetic | 2.1 | R2 | `IDENTITY` |
| Row `k` of `rk4_trajectory` is the state after `k · stride` steps, and row 0 is the initial state | 1 | R3 | `IDENTITY` |
| Shapes and dtype are as documented, and `jit` and `vmap` over states, over a batch of `Params` and over `dt` agree with the per-case result | 1 | R4 | `IDENTITY` |
| `n_out` and `stride` must be integers of at least 1, and a negative `dt` is accepted and follows `R` | 1, 2.2 | R5 | exact; `IDENTITY` for the formula |
| `solve_reference` has the documented signature, defaults, shape and dtype, and its first row is the initial state | 1 | S1 | `IDENTITY` |
| `t_eval` is validated, and a first time above zero works | 1 | S2 | exact, wrapper distance of D-044 |
| A failed solve raises `RuntimeError` | 1 | S3 | exact |
| A request below SciPy's `rtol` floor raises instead of being clamped, and the self-check tolerance `2.5e-14` is accepted | 4 | S4 | exact |
| Each closed-form solution satisfies its equation and initial condition | 3 | X1 to X4 | `IDENTITY` |
| RK4 converges at order 4 on all four exact problems, stays under the declared ceiling, and the wrapper is within the declared distance | 2.3, 3 | G1 | slope band, ceiling |
| RK4 converges at order 4 against the reference on regular pendulum states, outside the reference's noise floor | 2.3, 4 | G2 | slope band, validity rule |
| RK4 and DOP853 agree on chaotic-ish states over the declared horizon | 5 | G3 | worst case `≤` tolerance |
| RK4 and DOP853 agree on regular states over the declared horizon | 5 | G4 | worst case `≤` tolerance |
| RK4 and the wrapper agree on non-default parameters in units of `t_s` | 6 | G5 | worst case `≤` tolerance |
| Both trajectories conserve the energy of their own parameter set to the declared tolerance | 6 | G6 | worst case `≤` tolerance |
| A scheme that is fourth order on linear problems but second order in general (W2) passes the linear problems and fails the nonlinear ones; two schemes with a mistyped stage (W1, W3) fail every problem | 2.4 | N1 | slope band, ceiling |
| Swapping `L1` and `L2` in RK4 fails the wiring check | 6 | N2 | worst case `>` tolerance |
| Giving both integrators the same wrong parameters passes the agreement check and fails the energy anchor | 6 | N3 | both tolerances |
| The seeded cases have the declared counts, ranges and energies, and are reproducible from their streams | 5, 6 | C1 to C3 | exact |
| The register holds the declared values | 7 | K1 | exact |
| The check helpers fail on NaN and infinity, outside their limits and on a zero floor, and the slope helper recovers an exact power law | 7 | K2 | exact |

**What each family cannot catch.**

- **R1, R2 and R3** check one step and the indexing, not behaviour over a horizon. R2 is an independent implementation of the *same* scheme, so it cannot say whether the scheme is the right one; G1 does, through the order.
- **G1** proves the order without DOP853 or the pendulum. It is blind to pendulum-specific mistakes and to constant-factor errors below its ceilings.
- **G2** uses small amplitudes, where the pendulum is close to linear, so by itself it is weak against nonlinear order defects; the nonlinear problems of G1 carry that. Its validity rule stops it from passing at the reference's noise floor.
- **G3 and G4** are a backstop for the fast integrator on chaotic and regular motion with the default parameters. They are blind to defects below their tolerances and, with all parameters equal to one, to swapped parameters.
- **G5 and G6** catch dropped, swapped or unused parameters. They are blind to defects that change neither the trajectory at the check's resolution nor the energy.
- **Not covered here:** energy drift over `t = 1000`, time reversibility and the small-angle frequencies (milestone 4), and the Lyapunov estimators (milestone 5).
