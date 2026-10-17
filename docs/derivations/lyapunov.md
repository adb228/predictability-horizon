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

**The harmonic-oscillator check is also the owner's.** The owner gave `κ(V) = 2` and a bound of about `10⁻³` at `T = 1000`.

**The agent's re-check (2026-10-16).**
- I re-did steps 2 and 3 and agree with both. The bound holds for complex `V` with complex norms, and it holds for any scaling of the columns of `V`; the register declares one scaling (unit columns).
- **HO:** `A = [[0, 1], [−4, 0]]`, with eigenvectors `(1, ±2i)` and `V = [[1, 1], [2i, −2i]]`. Then `V* V = [[5, −3], [−3, 5]]`, with eigenvalues 8 and 2, so `κ(V) = √(8/2) = 2`. Both columns have the same norm, so scaling them does not change `κ`. The pair is the whole space, so the tie rule makes the overlap term 0, and `|λ(T)| ≤ ln 2 / T ≈ 6.9e-4` at `T = 1000`. This is D-080's `ln Ω / T` with `Ω = 2`.

### 1.2 Partial sums of the spectrum (agent's extension, for the owner to verify)

The owner mentioned this extension but did not derive it. QR's partial sum `S_k(T) = Σ_{i≤k} λᵢ(T)` is `(1/T) ln vol_k(e^{AT} Y_k)`, where `Y_k` is the first `k` columns of the orthonormal `frame0`, so `vol_k(Y_k) = 1`. This does not depend on the cadence, because the first `k` columns of the frame always span `e^{At} span(Y_k)`. Write `e^{AT} Y_k = V G C` with `G = e^{ΛT}` and `C = V⁻¹ Y_k`, an `n × k` matrix.

1. **Upper bound.** For matrices, `vol_k(XY) ≤ ‖∧ᵏX‖ vol_k(Y)`, and `‖∧ᵏX‖` is the product of the `k` largest singular values of `X`, which is at most `‖X‖ᵏ`. For the diagonal `G`, `‖∧ᵏG‖` is the product of its `k` largest moduli, `e^{S_k T}` with `S_k = a₁ + … + a_k`. So `vol_k(VGC) ≤ ‖V‖ᵏ e^{S_k T} vol_k(C) ≤ ‖V‖ᵏ ‖V⁻¹‖ᵏ e^{S_k T}`.
2. **Lower bound.** `vol_k(VGC) ≥ σ_min(V)ᵏ vol_k(GC)`, and by Cauchy–Binet, `vol_k(GC)² = Σ_I |det (GC)_I|² ≥ |det G_K|² |det C_K|²`, where `K = {1, …, k}` and `X_I` is the `k × k` submatrix on the rows `I`. Here `|det G_K| = e^{S_k T}`. Also `1 = vol_k(Y_k) = vol_k(VC) ≤ ‖V‖ᵏ vol_k(C)`. So `vol_k(VGC) ≥ κ(V)⁻ᵏ (|det C_K| / vol_k(C)) e^{S_k T}`.
3. **Combined:** `|S_k(T) − S_k| ≤ [k ln κ(V) + ln(vol_k(C) / |det C_K|)] / T`.

For `k = 1` this is the owner's bound, since `vol₁(C) = ‖c‖` and `|det C_K| = |c₁|`.
- The bound holds whether or not `K` splits a complex pair. In the complex eigenbasis `G` is diagonal and `C_K` is fixed in time. That is simpler than the real-block argument in D-080, which needed `k` not to split the block.
- `S₂` of L4 is therefore bounded too. It stays reported, not gated: D-080's gate set is unchanged.
- The confirming reviewer re-derived this bound and confirms it, valid over `ℂ` whether or not `K` splits the pair. D-080's old form, `[k ln κ + |ln γ_k|] / T` with unit columns, is also valid for `k = 1, 3, 4`, because `‖V‖ ≥ 1`. Neither form dominates the other, so the switch is a choice, not a correction. The owner checks this extension (question B of D-085).

### 1.3 The RK4 map

RK4 applied to `ẏ = A y` is the map `y ↦ R(hA) y = V R(hΛ) V⁻¹ y`, with the same `V`. After `N` steps, `G = diag(R(hμᵢ))ᴺ`, so every argument above holds with `aᵢ` replaced by the exact exponents of the map, `(1/h) ln|R(hμᵢ)|`. These are the references of D-080. Their ordering matches the ordering of the `Re μᵢ` for `h = 0.01`.

### 1.4 What is gated (unchanged from D-080, with the bound in the owner's form)

`T = 1000`, `h = 0.01`, no transient. The gates:
- `S₁`, `S₃` and `S₄` of L4, each within `B_k = [k ln κ(V) + ln(vol_k(C) / |det C_K|)] / T`, with `C = V⁻¹ frame0[:, :k]` and `B_k` computed in the test from the declared matrices and frame. Here **`V` is the complex eigenvector matrix of `A`, in unit columns, ordered by decreasing real part**. For the gated `k = 1, 3, 4` it gives the same `κ = 3` and the same volume ratios as D-080's real circulant `V`, because the change of basis within the pair is unitary; for `k = 2` it does not (confirming review, nit);
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

**Power (illustrative; corrected after the confirming review).** `N_Q` is a lab value and is not declared yet. For example, at `N_Q = 128`, `n = 4` and `κ_j = 10³`, the tolerance is about `1.1e-8` per interval, and at the cap `κ_cap = 10⁴` it is about `1.1e-7`. The Euler-frame control differs by about `h² tr(Df²)/2` per step. That is a scale, not a lower bound, because `tr(Df²)` changes sign for both systems (for Lorenz it is `108.1 + 20(28 − z) − 2x²`). The expected margin is about six orders of magnitude, but the lab's control run is what confirms it.

The analysis above also omits the rounding of `jacfwd`'s own `J` (about `s_t ε` per step at `κ(J) ≈ 1`), and Higham's QR constant carries an `mn` factor, so "small integers" is loose. Both are absorbed by counting the QR term `N_Q` times instead of once. **`c = 100` is a declared allowance, not a proved constant.**

## 3. The leading order of the per-step residual (D-079(b), re-review Rec 5; agent)

RK4's step map is `Ψ_h = φ_h + h⁵ E + O(h⁶)`, where `E(x)` is a fixed combination of elementary differentials of order 5. Then `DΨ_h = Dφ_h (I + h⁵ Dφ_h⁻¹ DE + O(h⁶))`, and since `Dφ_h = I + O(h)`,

`ln|det DΨ_h| − ln|det Dφ_h| = h⁵ tr DE(x) + O(h⁶) = h⁵ div E(x) + O(h⁶)`.

- **Lorenz and `(θ, ω)`:** `r = h⁵ a(x) + O(h⁶)`, with `a = div E`. For `(θ, ω)` there is an extra term, `+∇_θ ln det M · E_θ h⁵`, from using RK4's own `θ_{n+1}` (sign corrected after the confirming review), and it is of the same order. In general `a(x) ≠ 0`.
- **`(q, p)` linear, or near it:** for a Hamiltonian matrix `A`, `tr A^{2j+1} = 0`, and the linear part of `E` is a multiple of `A⁵ x`. So its divergence vanishes, and `r = O(h⁶)`. This is consistent with `R(z) R(−z) = 1 + z⁶/72 + z⁸/576` (the `z⁶` coefficient is `1/48 − 1/36 + 1/48 = +1/72`, which matches `|R(iy)|² = 1 − y⁶/72 + …`; sign corrected after the confirming review).
- **`(q, p)` nonlinear:** RK4 does not preserve volume for nonlinear Hamiltonian fields, so `div E` generally has a nonzero nonlinear part, and `r = O(h⁵)` with a coefficient that is small where the motion is close to linear. **Whether the `h⁵` coefficient dominates at the gate states cannot be settled on paper without evaluating it at those states**, which would be a look at the gate.

**Consequence: the rule of D-079 applies, and a separate `(q, p)` grid is declared a priori, `h ∈ {0.16, 0.08, 0.04, 0.02}`.** If `r` is `O(h⁶)` with the re-reviewer's estimate of about `2e-13` at `h = 0.01`, then at `h_min = 0.02` it is about `2e-13 · 2⁶ ≈ 1.3e-11`, about 58 times the floor of `2.2e-13`. The confirming reviewer estimates `(hμ)⁶/72 ≈ 5.6e-13` at `h = 0.01` for the pair at `|μ| ≈ 1.85`, which would give a margin of about 160; which estimate is right is not settled, and the decision is the same either way. If `r` is `O(h⁵)`, the margin is larger. The cost is `h_max ‖Df‖ ≈ 0.5`, above the `0.2` guideline. Section 4 shows that the slope stays above 4 in that range unless the leading coefficient nearly cancels. The `(θ, ω)` grid stays `{0.08, 0.04, 0.02, 0.01}` and the Lorenz grid stays `{0.005, …, 0.000625}`.

## 4. The slope near a cancellation (D-079(b), re-review Rec 4; agent, corrected after the confirming review). **Not clean.**

Write `r(h) = a hᵖ (1 + ρ h + O(h²))`, with `p = 5`, or `6` in the linear `(q, p)` case, and `ρ = b / a`. The least-squares slope of `log|r|` over the four grid points `h_max/8, h_max/4, h_max/2, h_max` is `p` plus the slope of `log|1 + ρh|`. Evaluated in this two-term model (paper arithmetic, not a run):

| `ρ h_max` | `−0.5` | `−0.8` | `−0.9` | `−0.95` | `−1` to `−8` | `−8.5` | `−20` | `+0.5` | `+3` |
|---|---|---|---|---|---|---|---|---|---|
| slope, `p = 5` | 4.71 | 4.31 | 4.01 | 3.70 | sign change | 7.23 | 6.22 | 5.16 | 5.51 |
| slope, `p = 6` | 5.71 | 5.31 | 5.01 | 4.70 | sign change | 8.23 | 7.22 | 6.16 | 6.51 |

- **Correct code fails to pass** only for `ρ h_max` roughly between `−8` and `−0.9` (`p = 5`), or between `−8` and `−0.99` (`p = 6`). Between `−8` and `−1` the zero of `1 + ρh` lies inside the grid and the sign rule fires (inconclusive). Just above `−1` the slope falls below 4.
- **Below `−8`**, all four points share a sign and the slope rises towards `p + 1`, so the gate passes. For positive `ρ` the slope only rises.

So, in this model, the band where correct code does not pass is about `|b| h_max / 8 < |a| < 1.1 |b| h_max`, with `a` and `b` of opposite signs. The first draft of this section said "fails exactly when `ρ h_max < −0.5`", which was wrong (confirming review, R2). The model is itself rough when `h ‖Df‖ ≈ 0.5`, and **I'm not certain of the exact edges.**

**The risk is not small for `(q, p)`.** For Lorenz and `(θ, ω)`, `a` has a linear part and generically vanishes only on a hypersurface, so the band is thin for small `h_max`, though that is not proved. In `(q, p)` the `h⁵` coefficient `a` has **no linear part** (section 3), while `b` does. At typical moving states `|a|` may therefore be comparable to `|b| h_max`, so `ρ h_max` of order 1 is plausible, on the coarse `(q, p)` grid in particular. A large fraction of the 8 `(q, p)` states could fall in the band. **The risk of a false fail there may be large, not residual.**

**The condition the owner put on Q6 does not come back clean.** The options for the owner:
1. **Accept the risk.** A failure at such a state is reported as a finding, with the state identified. This was the owner's first choice, made on the first draft's understatement.
2. **Symbolic analysis before the freeze.** Compute `a(x) / b(x)` from the elementary differentials of order 5 and 6 over the whole sampling box, never at a gate state. If the band is not empty, the owner decides between a new a-priori grid and option 1. Mapping the band densely comes close to the spirit of the Q8 refusal, and the owner should weigh that.
3. **A slope over the three finer points only.** This is less exposed at the coarse end, but it has less leverage against the floor.
4. **An aggregate statistic** (agent's recommendation for `(q, p)`; for Lorenz and `(θ, ω)` as well, if the owner prefers one rule). Fit the slope of `log max_states |r(h)|`, the largest `|r|` over the 8 states at each `h`, and keep the floor on that maximum.
   - The maximum is set by the states without a near-cancellation, so it scales like `hᵖ` unless the states that set the maximum themselves near-cancel (not proved; see D-086 for a two-term example at the threshold).
   - A wrong formula leaves an `O(h)` residual at every state, so the aggregate slope stays near 1 and the gate still fails. The controls of D-079 test exactly that.
   - The cost is that a defect present at only a few states, with the others correct, could be masked. A wrong `ℓ` formula does not behave that way.
   - Per-state slopes and signs are then reported, not gated.

**The owner's choice (D-086, 2026-10-17): option 4, for all three systems.** The negative controls of D-079 must still fail under the aggregate rule.

## 5. `δ_BQ`: the worst-case and the probabilistic forms (D-082, Q7; agent, for the owner to verify)

Benettin's separation over an interval of `N_B` steps obeys `δ_{m+1} = J_m δ_m + e_m + O(d0²)`. Here `e_m` is the rounding injected at step `m`. The two trajectories are each rounded at a size of order `ε X`, `X = max_abs_state`, so their difference carries at most `|e_m| ≤ s ε X`. The renormalization `x̃ ← x + (d0/d) Δ` injects one more `ε X`.

- **Per-step constant.** One RK4 step is `x + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)`. Its rounding is dominated by the final additions, at most about `2 ε X` for each trajectory, plus `O(h ε X)`. So `s = 5` covers the two trajectories and the renormalization.
- **Worst case.** All injections have the same sign, and an injected error is amplified no more than the separation itself over the rest of the interval. The relative error per interval is then at most `(s ε X / d0) · N_B`. Divided by `τ_B = N_B h`, this gives **`δ_BQ^wc = s ε X / (d0 h)`**, independent of `λ` and of `τ_B`. With alignment and `λ > 0`, the factor `N_B` could be replaced by about `1/(λh)`. That is not used, because alignment is not a theorem on every state.

  **Two assumptions, neither proved over a finite interval:**
  - injected errors are amplified no more than the separation itself, which holds once the separation is aligned with the leading direction;
  - `|δ_m| ≥ d0` throughout the interval, which bounding `|e_m| / |δ_m|` by `s ε X / d0` needs. On regular states the separation can dip below `d0` within an interval (confirming review, Rec 3).

  **I'm not certain of either.**
- **Probabilistic (reported only).** Model the `e_m` as independent, zero-mean, of size `s ε X`. The per-interval relative error then has a standard deviation of about `(s ε X / d0) √N_B`, and the `n_renorm` intervals average down by `√n_renorm`. This gives `δ_BQ^pr = z · s ε X / (d0 h √(N_total))`, where `N_total` is the number of steps in the window and `z = 5`, a false-fail rate of about `6e-7` per state under the model. It is a model, not a theorem, and it is reported only.

**Values on paper**, `s = 5`, `d0 = 1e-9`:

| System | `h` | `X` | `δ_BQ^wc` | `δ_BQ^pr` | `|λ₁|` |
|---|---|---|---|---|---|
| Lorenz | `1/128` | up to 78 (section 6) | up to `1.1e-2` | about `5e-5` | 0.906 |
| pendulum, non-rotating | 0.01 | typically about 3.2; up to about 6 (`|q| ≤ π`, `|p| ≤ √(2 · 6.03 · tr M) ≈ 6.0`) | `3.5e-4` to about `6.7e-4` | about `6e-6` to `1e-5` | order 0.1 to 1 |
| pendulum, rotating | 0.01 | typically `10³`; at most the cap, `6.6 · 10³` | typically `0.11`; up to `0.73` | about `2e-3` to `1.2e-2` | order 0.1 to 1 |

**Where the gate has little power (stated, as the owner asked).**
- **Rotating pendulum states:** `δ_BQ^wc` is as large as `λ₁` itself or larger, so the gate cannot tell a correct estimator from a wrong one there.
- **Lorenz:** `δ_BQ^wc` is about 1 percent of `λ₁`. That catches any factor or sign error, but not a bias below 1 percent; the Lorenz literature gate (D-081) covers that range.
- **Non-rotating pendulum states:** the gate has power down to about `3.5e-4` to `6.7e-4`.
- **The coordinate-mismatch control of D-082** may not fail under the worst-case gate. Its term is of order `ln κ(DΦ) / T ≲ ln 6.85 / 1000 ≈ 1.9e-3`: only a few times `δ_BQ^wc` on non-rotating states, and far below it on rotating ones. If it does not fail, the one rule for every control applies: reported to the owner, kept as a finding, nothing tuned (confirming review, Rec 5).

The script marks every state where `δ_BQ^wc > 0.1 |λ̂_{QR,1}|` as "low power". That is a reporting mark, not a gate.

**Correction to the owner's expectation.** The `6e-4` quoted for Lorenz in D-082 used `s ≈ 1`, `X ≈ 50` and the aligned factor `1/(λh) ≈ 141`. The worst-case `1.1e-2` uses `s = 5`, the cap `X = 78` and `N_B = 320`. The ratio, about `5 × 1.56 × 2.27 ≈ 18`, has three causes, and alignment is only about 2.3 of it (confirming review, Rec 1). The conclusion "lean on Lorenz" still holds, with that number. **D-083's sentence that `δ_BQ` scales with `1/τ_B` is superseded:** the worst-case form does not depend on `τ_B`.

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

  `V` is convex, so its maximum over the solid ellipsoid is on the boundary. The gate's initial conditions satisfy `V(0) ≤ 15² + 15² + 28² = 1234 < R² ≈ 1540`, so the trajectory stays in the ball, where `|x|, |y| ≤ R` and `z ≤ σ + ρ + R ≈ 77.25`. **`X_cap = 1.01 · (σ + ρ + R) ≈ 78.0`.** The factor 1.01 is a declared allowance for RK4's departure from the exact flow, not a derived constant. The attractor itself has `X` of about 48.
- **L4 and HO: no cap, because no agreement gate applies to them** (D-080 gates their Benettin against `B₁`, not against QR; confirming review, Rec 2). For the record: L4 has `X = 0`, where the formula `δ_BQ^wc` would be exactly 0, a certain false fail if an agreement gate were ever added. HO's supremum is `max(1, Ω) · √(x₀² + v₀²/Ω²) = 2 √1.0625 ≈ 2.0616`. Wherever a cap is used, the register stores the formula, not a rounded display.

## 7. Summary for the owner

| Owed derivation | Result |
|---|---|
| D-080, partial sums | Your `k = 1` bound, verified. The agent's extension to `S_k` (section 1.2) is for you to check. |
| D-079(a), constant `c` | `c = 100`, an allowance, with one unproved assumption about the intermediate conditioning (section 2). |
| Rec 5, the leading order in `(q, p)` | Settled by the pre-agreed rule: a separate `(q, p)` grid `{0.16, 0.08, 0.04, 0.02}` (section 3). |
| Rec 4, the slope near a cancellation | **Not clean.** Corrected after the confirming review: the false-fail band is `−8 ≲ ρ h_max ≲ −0.9`, and in `(q, p)` the risk may be large, not residual. Four options for the owner (section 4). |
| `δ_BQ` | The worst-case form `s ε X / (d0 h)` with `s = 5`, resting on two unproved assumptions; the probabilistic form is reported. Lorenz is about `1.1e-2`, not the `6e-4` quoted earlier (section 5). |
| `X` caps | Pendulum from energy conservation; Lorenz from the trapping ball `R ≈ 39.25`; none for L4 and HO, which have no agreement gate (section 6). |
