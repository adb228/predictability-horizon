# Lyapunov estimators: the derivations owed before the freeze

This note will hold the full derivation and the claim-to-test table for milestone 5 (D-076). **This first part contains only the derivations that the design (D-076 to D-085) says must be done before the register is frozen.** Each section says who derived it, and what is still uncertain. Nothing here was measured. Every number is computed on paper from declared values, and none comes from a run of the estimators. The owner checks every line before anything is frozen (D-039).

Notation:
- `h` is the step, `ε = ε_mach = 2⁻⁵²`, and `n` is the dimension.
- `R(z) = 1 + z + z²/2 + z³/6 + z⁴/24` is RK4's stability polynomial.
- For a matrix, `‖·‖` is the 2-norm, `σ_min` its smallest singular value, and `κ(V) = ‖V‖ ‖V⁻¹‖`.
- `vol_k(Y)` is the `k`-volume spanned by the columns of the `n × k` matrix `Y`, `√det(Y* Y)`.

## 1. The finite-time bound for linear systems (D-080)

### 1.1 The leading exponent (derived by the owner, 2026-10-16)

Take `ẋ = A x` with `A` diagonalizable: `A = V Λ V⁻¹`, eigenvalues `μᵢ` and real parts `aᵢ = Re μᵢ`, sorted so that `a₁ ≥ a₂ ≥ …`. The claim is that the finite-time exponent `λ(T) = (1/T) ln(‖v(T)‖ / ‖v₀‖)` converges to `a₁` with an error of order `1/T`.

1. **The solution.** `v(T) = e^{AT} v₀ = V e^{ΛT} V⁻¹ v₀`. In the eigenbasis the system decouples, and each coordinate grows like `e^{μᵢ T}`.
2. **Upper bound.** `‖v(T)‖ ≤ ‖V‖ ‖e^{ΛT}‖ ‖V⁻¹‖ ‖v₀‖ = κ(V) e^{a₁T} ‖v₀‖`, because `e^{ΛT}` is diagonal and its norm is its largest modulus, `e^{a₁T}`. So `λ(T) ≤ a₁ + ln κ(V) / T`.
3. **Lower bound.** Let `c = V⁻¹ v₀`. Then `‖v(T)‖ = ‖V e^{ΛT} c‖ ≥ σ_min(V) |c₁| e^{a₁T}`, and `‖v₀‖ = ‖V c‖ ≤ ‖V‖ ‖c‖`. With `‖V‖ / σ_min(V) = κ(V)`, this gives `λ(T) ≥ a₁ − [ln κ(V) + ln(‖c‖ / |c₁|)] / T`.
4. **Combined:** `|λ(T) − a₁| ≤ [ln κ(V) + ln(‖c‖ / |c₁|)] / T`.

The bias is of order `1/T` and has two sources:
- `κ(V)` measures how far the eigenvectors are from perpendicular; it is 1 for a normal `A`.
- `‖c‖ / |c₁|` measures how little the starting vector points along the leading direction.

**Ties.** When the leading moduli tie, for example in a complex pair with the same real part, `|c₁|` is replaced by the norm of the starting vector's component in the tied group, since every entry of that group has `|e^{μT}| = e^{a₁T}`.

**Diagonalizability is required.** A Jordan block of size `m` adds a factor `t^{m−1}` and an error of order `(m − 1) ln T / T`. **L4 is diagonalizable:** its eigenvalues `0.5`, `−0.2 ± i` and `−1` are distinct.

**Agent's check (2026-10-16).**
- I re-did steps 2 and 3 and agree with both. The bound holds for complex `V` with complex norms, and it holds for any scaling of the columns of `V`; the register declares one scaling (unit columns).
- **HO:** `A = [[0, 1], [−4, 0]]`, with eigenvectors `(1, ±2i)` and `V = [[1, 1], [2i, −2i]]`. Then `V* V = [[5, −3], [−3, 5]]`, with eigenvalues 8 and 2, so `κ(V) = √(8/2) = 2`. The pair is the whole space, so the tie rule makes the overlap term 0, and `|λ(T)| ≤ ln 2 / T ≈ 6.9e-4` at `T = 1000`. This is D-080's `ln Ω / T` with `Ω = 2`.

### 1.2 Partial sums of the spectrum (agent's extension, for the owner to verify)

The owner mentioned this extension but did not derive it. QR's partial sum `S_k(T) = Σ_{i≤k} λᵢ(T)` is `(1/T) ln vol_k(e^{AT} Y_k)`, where `Y_k` is the first `k` columns of the orthonormal `frame0`, so `vol_k(Y_k) = 1`. This does not depend on the cadence, because the first `k` columns of the frame always span `e^{At} span(Y_k)`. Write `e^{AT} Y_k = V G C` with `G = e^{ΛT}` and `C = V⁻¹ Y_k`, an `n × k` matrix.

1. **Upper bound.** For matrices, `vol_k(XY) ≤ ‖∧ᵏX‖ vol_k(Y)`, and `‖∧ᵏX‖` is the product of the `k` largest singular values of `X`, which is at most `‖X‖ᵏ`. For the diagonal `G`, `‖∧ᵏG‖` is the product of its `k` largest moduli, `e^{S_k T}` with `S_k = a₁ + … + a_k`. So `vol_k(VGC) ≤ ‖V‖ᵏ e^{S_k T} vol_k(C) ≤ ‖V‖ᵏ ‖V⁻¹‖ᵏ e^{S_k T}`.
2. **Lower bound.** `vol_k(VGC) ≥ σ_min(V)ᵏ vol_k(GC)`, and by Cauchy–Binet, `vol_k(GC)² = Σ_I |det (GC)_I|² ≥ |det G_K|² |det C_K|²`, where `K = {1, …, k}` and `X_I` is the `k × k` submatrix on the rows `I`. Here `|det G_K| = e^{S_k T}`. Also `1 = vol_k(Y_k) = vol_k(VC) ≤ ‖V‖ᵏ vol_k(C)`. So `vol_k(VGC) ≥ κ(V)⁻ᵏ (|det C_K| / vol_k(C)) e^{S_k T}`.
3. **Combined:** `|S_k(T) − S_k| ≤ [k ln κ(V) + ln(vol_k(C) / |det C_K|)] / T`.

For `k = 1` this is the owner's bound, since `vol₁(C) = ‖c‖` and `|det C_K| = |c₁|`.
- The bound holds whether or not `K` splits a complex pair. In the complex eigenbasis `G` is diagonal and `C_K` is fixed in time. That is simpler than the real-block argument in D-080, which needed `k` not to split the block.
- `S₂` of L4 is therefore bounded too. It stays reported, not gated: D-080's gate set is unchanged.
- **I'm not certain I have every step right; the owner checks this one.** One reviewer re-derived a version of it in the real block form.

### 1.3 The RK4 map

RK4 applied to `ẏ = A y` is the map `y ↦ R(hA) y = V R(hΛ) V⁻¹ y`, with the same `V`. After `N` steps, `G = diag(R(hμᵢ))ᴺ`, so every argument above holds with `aᵢ` replaced by the exact exponents of the map, `(1/h) ln|R(hμᵢ)|`. These are the references of D-080. Their ordering matches the ordering of the `Re μᵢ` for `h = 0.01`.

### 1.4 What is gated (unchanged from D-080, with the bound in the owner's form)

`T = 1000`, `h = 0.01`, no transient. The gates:
- `S₁`, `S₃` and `S₄` of L4, each within `B_k = [k ln κ(V) + ln(vol_k(C) / |det C_K|)] / T`, with `C = V⁻¹ frame0[:, :k]`, `V` in unit columns, and `B_k` computed in the test from the declared matrices and frame;
- `S₁` of HO, within `ln κ(V) / T = ln 2 / T`;
- Benettin's `λ̂` on both systems, within `B₁`, with `c = V⁻¹ direction0`;
- the full sums by the identity of section 2.

## 2. The rounding bound of the bookkeeping identity (D-079(a); agent, for the owner to verify)

For an interval `j` of `N_Q = steps_per_qr` steps, the QR side computes `Σᵢ ln|R_ii| = ln|det Y_end|`, with `Y_end = J_{N_Q} ⋯ J₁ Q_j`. `log_det_steps[j]` is `Σ_m ln|det J_m|`. In exact arithmetic the two are equal. The rounding has three sources, each bounded to first order by `δ ln|det Y| = Re tr(Y⁻¹ δY)`, so `|δ ln|det Y|| ≤ n ‖Y⁻¹‖ ‖δY‖ = n κ(Y) ‖δY‖ / ‖Y‖`:

1. **Tangent steps.** Each `jvp` step computes `J_m Y_m` with a relative backward error of at most `s_t ε`, where `s_t` is of the order of the number of stage operations. Its contribution is at most `n κ(Y_m) s_t ε`.
2. **The QR at the end of the interval.** Householder QR is backward stable, with `‖δY‖ ≤ c_q n ε ‖Y‖`. Its contribution is at most `n κ(Y_end) c_q n ε`.
3. **`slogdet` at every step.** LU with partial pivoting has a backward error of at most about `c_l n ε ‖J_m‖`, and `J_m = I + O(h)` is well conditioned. Its contribution is at most about `c_l n² ε` per step.

**Bound.** Write `κ_j` for `κ₂(R_j) = κ(Y_end)` and assume `κ(Y_m) ≤ κ_j` within the interval. The total is at most `ε N_Q n κ_j (s_t + n c_q + n c_l)`. Higham's backward-error constants for these operations at `n ≤ 4` are small integers, so `s_t + n c_q + n c_l ≤ 100` is a generous allowance. **The declared tolerance per interval is `100 · ε · N_Q · n · κ_j`, so `c = 100`.**

**Uncertain:**
- The assumption `κ(Y_m) ≤ κ_j` for the intermediate frames is not proved. The conditioning of a partial propagator is usually monotone over a short interval, but not necessarily. Recording `κ` at every step would remove the assumption, at the cost of an SVD per step.
- The constant 100 is an allowance, not a proved constant. **I'm not certain of either.**

**Power.** With `N_Q ≤ 128`, `n = 4` and `κ_j ≤ 10³`, the tolerance is at most about `1.1e-8` per interval. The Euler-frame control differs by `O(h²) ‖Df‖²` per step, `1e-4` or more per step for both systems, so it fails by orders of magnitude.

## 3. The leading order of the per-step residual (D-079(b), re-review Rec 5; agent)

RK4's step map is `Ψ_h = φ_h + h⁵ E + O(h⁶)`, where `E(x)` is a fixed combination of elementary differentials of order 5. Then `DΨ_h = Dφ_h (I + h⁵ Dφ_h⁻¹ DE + O(h⁶))`, and since `Dφ_h = I + O(h)`,

`ln|det DΨ_h| − ln|det Dφ_h| = h⁵ tr DE(x) + O(h⁶) = h⁵ div E(x) + O(h⁶)`.

- **Lorenz and `(θ, ω)`:** `r = h⁵ a(x) + O(h⁶)`, with `a = div E`. For `(θ, ω)` there is an extra term, `−∇_θ ln det M · E_θ h⁵`, from using RK4's own `θ_{n+1}`, and it is of the same order. In general `a(x) ≠ 0`.
- **`(q, p)` linear, or near it:** for a Hamiltonian matrix `A`, `tr A^{2j+1} = 0`, and the linear part of `E` is a multiple of `A⁵ x`. So its divergence vanishes, and `r = O(h⁶)`. This is consistent with `R(z) R(−z) = 1 − z⁶/72 + …`.
- **`(q, p)` nonlinear:** RK4 does not preserve volume for nonlinear Hamiltonian fields, so `div E` generally has a nonzero nonlinear part, and `r = O(h⁵)` with a coefficient that is small where the motion is close to linear. **Whether the `h⁵` coefficient dominates at the gate states cannot be settled on paper without evaluating it at those states**, which would be a look at the gate.

**Consequence: the rule of D-079 applies, and a separate `(q, p)` grid is declared a priori, `h ∈ {0.16, 0.08, 0.04, 0.02}`.** If `r` is `O(h⁶)` with the re-reviewer's estimate of about `2e-13` at `h = 0.01`, then at `h_min = 0.02` it is about `2e-13 · 2⁶ ≈ 1.3e-11`, about 58 times the floor of `2.2e-13`. If `r` is `O(h⁵)`, the margin is larger. The cost is `h_max ‖Df‖ ≈ 0.5`, above the `0.2` guideline. Section 4 shows that the slope stays above 4 in that range unless the leading coefficient nearly cancels. The `(θ, ω)` grid stays `{0.08, 0.04, 0.02, 0.01}` and the Lorenz grid stays `{0.005, …, 0.000625}`.

## 4. The slope near a cancellation (D-079(b), re-review Rec 4; agent). **Not clean.**

Write `r(h) = a hᵖ (1 + ρ h + O(h²))`, with `p = 5`, or `6` in the linear `(q, p)` case, and `ρ = b / a`. The least-squares slope of `log|r|` over a grid from `h_max/8` to `h_max` is `p` plus the slope of `log|1 + ρh|`:
- **`ρ h_max ≥ −0.5`:** that slope stays within about `±0.5`, so the measured slope is at least about `p − 0.5 ≥ 4.5`. The gate passes.
- **`−1 < ρ h_max < −0.5`:** `1 + ρh` is small at the coarse end, the slope falls, and it can fall below 4. The gate **fails correct code**.
- **`ρ h_max ≤ −1`:** `r` changes sign on the grid, the sign rule fires, and the test fails as inconclusive.

So a correct implementation fails at a state exactly when `ρ(x) h_max < −0.5`, that is, when the leading coefficient `a(x)` is within `2 |b(x)| h_max` of zero. `a` is a smooth function of the state that vanishes on a hypersurface. The chance that a random gate state lands in that band is small for small `h_max`, but **it cannot be bounded on paper** without evaluating `a` and `b`. Evaluating them at the gate states would be a look at the gate.

**So the condition the owner put on Q6 does not come back clean for this derivation.** The options are for the owner:
1. Accept the residual risk. A failure at such a state is reported as a finding, never answered by changing the threshold, the grid or the states. This is consistent with Q8.
2. Before the freeze, run a symbolic analysis of `a(x) / b(x)` over the whole sampling box, from the elementary differentials of order 5 and 6 (SymPy is already the cross-check tool of milestone 2). This bounds the band from above **without** evaluating any gate state, and it is more work.

## 5. `δ_BQ`: the worst-case and the probabilistic forms (D-082, Q7; agent, for the owner to verify)

Benettin's separation over an interval of `N_B` steps obeys `δ_{m+1} = J_m δ_m + e_m + O(d0²)`. Here `e_m` is the rounding injected at step `m`. The two trajectories are each rounded at a size of order `ε X`, `X = max_abs_state`, so their difference carries at most `|e_m| ≤ s ε X`. The renormalization `x̃ ← x + (d0/d) Δ` injects one more `ε X`.

- **Per-step constant.** One RK4 step is `x + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)`. Its rounding is dominated by the final additions, at most about `2 ε X` for each trajectory, plus `O(h ε X)`. So `s = 5` covers the two trajectories and the renormalization.
- **Worst case.** All injections have the same sign, and an injected error is amplified no more than the separation itself over the rest of the interval. The relative error per interval is then at most `(s ε X / d0) · N_B`. Divided by `τ_B = N_B h`, this gives **`δ_BQ^wc = s ε X / (d0 h)`**, independent of `λ` and of `τ_B`. With alignment and `λ > 0`, the factor `N_B` could be replaced by about `1/(λh)`. That is not used, because alignment is not a theorem on every state.

  **Uncertain:** the assumption that injected errors are amplified no more than the separation holds once the separation is aligned with the leading direction, but it is not proved over a finite interval, and **I'm not certain of it.**
- **Probabilistic (reported only).** Model the `e_m` as independent, zero-mean, of size `s ε X`. The per-interval relative error then has a standard deviation of about `(s ε X / d0) √N_B`, and the `n_renorm` intervals average down by `√n_renorm`. This gives `δ_BQ^pr = z · s ε X / (d0 h √(N_total))`, where `N_total` is the number of steps in the window and `z = 5`, a false-fail rate of about `6e-7` per state under the model. It is a model, not a theorem, and it is reported only.

**Values on paper**, `s = 5`, `d0 = 1e-9`:

| System | `h` | `X` | `δ_BQ^wc` | `δ_BQ^pr` | `|λ₁|` |
|---|---|---|---|---|---|
| Lorenz | `1/128` | up to 78 (section 6) | up to `1.1e-2` | about `5e-5` | 0.906 |
| pendulum, non-rotating | 0.01 | about 3.2 | about `3.5e-4` | about `6e-6` | order 0.1 to 1 |
| pendulum, rotating | 0.01 | `10³` to `6.6 · 10³` | `0.11` to `0.73` | about `2e-3` to `1.2e-2` | order 0.1 to 1 |

**Where the gate has little power (stated, as the owner asked).**
- **Rotating pendulum states:** `δ_BQ^wc` is as large as `λ₁` itself or larger, so the gate cannot tell a correct estimator from a wrong one there.
- **Lorenz:** `δ_BQ^wc` is about 1 percent of `λ₁`. That catches any factor or sign error, but not a bias below 1 percent; the Lorenz literature gate (D-081) covers that range.
- **Non-rotating pendulum states:** the gate has power down to about `3.5e-4`.

The script marks every state where `δ_BQ^wc > 0.1 |λ̂_{QR,1}|` as "low power". That is a reporting mark, not a gate.

**Correction to the owner's expectation.** The `6e-4` quoted for Lorenz in D-082 used the aligned form. The worst-case form is about 20 times larger, about `1.1e-2`, at the trapping-ball `X`. The conclusion "lean on Lorenz" still holds, with that number.

## 6. The caps on `X` (the owner's follow-up to Q7; agent, for the owner to verify)

Above the cap the agreement gate fails as inconclusive, so a defect that inflates the state cannot loosen `δ_BQ`. Each cap is derived from the physics and the declared values, never from a run.

- **Pendulum, per state** (default parameters, `E_s = 3`). `H = E(0)` is conserved by the flow; RK4 drifts by far less than `0.01 E_s` over the window (D-055). So the kinetic energy is at most `K_max = E(0) + E_s + 0.01 E_s`.
  - **Momenta:** `‖p‖² = ωᵀ M² ω ≤ λ_max(M) · 2K ≤ tr M · 2K`, so `|p_i| ≤ √(2 K_max tr M)`, where `tr M = (m₁ + m₂) L₁² + m₂ L₂² = 3`.
  - **Angular speed:** `‖ω‖² ≤ 2K / λ_min(M)`, with `λ_min(M) ≥ det M / tr M ≥ m₁ m₂ L₁² L₂² / tr M = 1/3`.
  - **Angles:** `|q_i(t)| ≤ |q_i(0)| + t ω_max`, with `t ≤ t_trans + T = 1100`.
  - **Cap:** `X_cap = max(|q(0)|_∞ + 1100 ω_max, p_max)`. For a rest state with `E(0) ≤ 3`, `ω_max ≤ √(2 · 6.03 · 3) ≈ 6.0`, so `X_cap ≤ π + 6600 ≈ 6.6 · 10³`.

  This cap catches a blow-up that breaks energy conservation. It does not catch an inflation within the physical range, and on rotating states the gate has little power anyway (section 5).
- **Lorenz.** Let `V(x, y, z) = x² + y² + (z − σ − ρ)²`. Along the flow, `V̇ = −2σx² − 2y² − 2β z² + 2β(σ + ρ) z`, which is negative outside the ellipsoid `σx² + y² + β(z − (σ+ρ)/2)² = β(σ+ρ)²/4`.

  The maximum of `V` on that ellipsoid, for `σ ≥ 1` and `β ≥ 2`, is `R² = β²(σ+ρ)² / (4(β − 1))`, so `R = 38 · (8/3) / (2√(5/3)) ≈ 39.25`. (With `x = 0` the maximum is over `z`; the stationary point lies inside the ellipsoid's range because `β ≥ 2`.) Every ball `V ≤ R'²` with `R' ≥ R` is therefore forward invariant.

  The gate's initial conditions satisfy `V(0) ≤ 15² + 15² + 28² = 1234 < R²`, so the trajectory stays in the ball, where `|x|, |y| ≤ R` and `z ≤ σ + ρ + R ≈ 77.25`. **`X_cap = 1.01 · 77.25 ≈ 78.0`**; the 1 percent allows for RK4's departure from the exact flow.
- **L4:** `state0 = 0`, so `X = 0` and `δ_BQ` reduces to the renormalization rounding at `d0`.
- **HO:** energy gives `X ≤ max(1, Ω) · √(x₀² + v₀²/Ω²) = 2 · √(1 + 0.0625) ≈ 2.06`.

## 7. Summary for the owner

| Owed derivation | Result |
|---|---|
| D-080, partial sums | Your `k = 1` bound, verified. The agent's extension to `S_k` (section 1.2) is for you to check. |
| D-079(a), constant `c` | `c = 100`, an allowance, with one unproved assumption about the intermediate conditioning (section 2). |
| Rec 5, the leading order in `(q, p)` | Settled by the pre-agreed rule: a separate `(q, p)` grid `{0.16, 0.08, 0.04, 0.02}` (section 3). |
| Rec 4, the slope near a cancellation | **Not clean.** A residual false-fail risk that cannot be bounded on paper without the symbolic analysis of option 2 (section 4). |
| `δ_BQ` | The worst-case form `s ε X / (d0 h)` with `s = 5`; the probabilistic form is reported. Lorenz is about `1.1e-2`, not the `6e-4` quoted earlier (section 5). |
| `X` caps | Pendulum from energy conservation; Lorenz from the trapping ball `R ≈ 39.25` (section 6). |
