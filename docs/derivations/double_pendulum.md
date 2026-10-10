# Double pendulum: equations of motion

This note derives the equations the simulator implements and states every claim that a test checks. Section 9 maps each claim to the test that guards it. Decisions behind the tests are in `docs/DECISIONS.md` (D-028 to D-034).

## 1. Setup, conventions and units

Two point masses `m1`, `m2` hang from massless rigid rods of lengths `L1`, `L2` in a vertical plane under uniform gravity `g`, with no friction. The first rod is pinned at the origin.

- `θ1` is the angle of rod 1 from the **downward vertical**. `θ2` is the angle of rod 2 from the downward vertical too: it is **absolute**, not measured relative to rod 1. The brief's energy formulas assume this convention (checked by O1).
- `ωi = θ̇i`, `αi = ω̇i`. The state is `s = (θ1, θ2, ω1, ω2)`, an array of shape `(4,)`. Angles are in radians and are **never wrapped** when integrated (D-010); section 5 shows why that is safe.
- Defaults are nondimensional: `m1 = m2 = L1 = L2 = g = 1`. Section 8 says what that choice means.

## 2. Energy from geometry

Put `y` upward. The positions are

$$x_1 = L_1 \sin\theta_1,\quad y_1 = -L_1\cos\theta_1,\qquad x_2 = x_1 + L_2\sin\theta_2,\quad y_2 = y_1 - L_2\cos\theta_2 .$$

Differentiating, `|v1|² = L1² ω1²` and, using `cos θ1 cos θ2 + sin θ1 sin θ2 = cos(θ1 − θ2)`,

$$|v_2|^2 = L_1^2\omega_1^2 + L_2^2\omega_2^2 + 2L_1L_2\,\omega_1\omega_2\cos(\theta_1-\theta_2).$$

With `Δ = θ1 − θ2`, the kinetic and potential energies are

$$T = \tfrac12 m_1|v_1|^2 + \tfrac12 m_2|v_2|^2 = \tfrac12(m_1+m_2)L_1^2\omega_1^2 + \tfrac12 m_2L_2^2\omega_2^2 + m_2L_1L_2\,\omega_1\omega_2\cos\Delta,$$

$$V = m_1 g\,y_1 + m_2 g\,y_2 = -(m_1+m_2)\,g L_1\cos\theta_1 - m_2\,g L_2\cos\theta_2 .$$

These are exactly the brief's `T` and `V`. In matrix form `T = ½ ωᵀ M(θ) ω` with the **mass matrix**

$$M(\theta) = \begin{pmatrix} (m_1+m_2)L_1^2 & m_2L_1L_2\cos\Delta \\ m_2L_1L_2\cos\Delta & m_2L_2^2 \end{pmatrix}.$$

Its determinant factors:

$$\det M = (m_1+m_2)m_2L_1^2L_2^2 - m_2^2L_1^2L_2^2\cos^2\Delta = L_1^2L_2^2\,m_2\,\bigl(m_1 + m_2\sin^2\Delta\bigr).$$

Since `M11 > 0` and `det M > 0` whenever `m1 > 0` and `m2 > 0`, `M` is symmetric positive definite and always invertible, for every `θ`.

## 3. Euler-Lagrange equations

With `ℒ = T − V`, the equations are `d/dt(∂ℒ/∂ωi) − ∂ℒ/∂θi = 0`.

- `∂ℒ/∂ω = M ω`, so `d/dt(∂ℒ/∂ω) = M α + Ṁ ω`, where `Ṁ = Σk (∂k M) ωk`.
- `M` depends on `θ` only through `Δ`, so `∂1 M = M′` and `∂2 M = −M′` with `M′ = [[0, −c sin Δ], [−c sin Δ, 0]]` and `c = m2 L1 L2`. Hence `Ṁ = M′ (ω1 − ω2)`.
- `∂T/∂θi = ½ ωᵀ(∂i M) ω`, which gives `∂T/∂θ1 = −c ω1 ω2 sin Δ` and `∂T/∂θ2 = +c ω1 ω2 sin Δ`.
- `∂V/∂θ1 = (m1+m2) g L1 sin θ1` and `∂V/∂θ2 = m2 g L2 sin θ2`.

Component 1 of `M α + Ṁ ω − ∂T/∂θ + ∂V/∂θ = 0`. The velocity terms are

$$(\dot M\omega)_1 - \partial_{\theta_1}T = -c\sin\Delta\,(\omega_1-\omega_2)\omega_2 + c\sin\Delta\,\omega_1\omega_2 = c\sin\Delta\;\omega_2^2 ,$$

and component 2 is

$$(\dot M\omega)_2 - \partial_{\theta_2}T = -c\sin\Delta\,(\omega_1-\omega_2)\omega_1 - c\sin\Delta\,\omega_1\omega_2 = -c\sin\Delta\;\omega_1^2 .$$

So the equations of motion are

$$\boxed{M(\theta)\,\alpha = b(\theta,\omega)},\qquad
\begin{aligned}
b_1 &= -\,m_2L_1L_2\sin\Delta\;\omega_2^2 - (m_1+m_2)\,gL_1\sin\theta_1,\\
b_2 &= +\,m_2L_1L_2\sin\Delta\;\omega_1^2 - m_2\,gL_2\sin\theta_2 .
\end{aligned}$$

Inverting the `2 × 2` system gives the closed form

$$\alpha_1 = \frac{M_{22}b_1 - M_{12}b_2}{\det M},\qquad \alpha_2 = \frac{M_{11}b_2 - M_{12}b_1}{\det M}.$$

Either form (solve `M α = b` numerically, or use the closed form) is a valid implementation. The simulator integrates the autonomous first-order system

$$\dot s = f(s) = (\omega_1,\ \omega_2,\ \alpha_1,\ \alpha_2).$$

## 4. Energy is conserved

Let `E = T + V`. Along any solution,

$$\dot E = \omega^\top M\alpha + \tfrac12\omega^\top\dot M\omega + \nabla V\cdot\omega .$$

The Euler-Lagrange equation says `M α = ∂T/∂θ − ∇V − Ṁ ω`. Substituting,

$$\dot E = \omega^\top\partial_\theta T - \omega^\top\dot M\omega + \tfrac12\omega^\top\dot M\omega = \omega^\top\partial_\theta T - \tfrac12\omega^\top\dot M\omega .$$

But `ωᵀ ∂θT = Σi ωi · ½ ωᵀ(∂i M)ω = ½ ωᵀ(Σi ωi ∂i M)ω = ½ ωᵀ Ṁ ω`, so `Ė = 0`.

In code terms, `∇E · f = 0` **at every state**, with no integration needed. That is what makes it a cheap, sharp test of the equations of motion against the energy.

## 5. Symmetries

All three follow from the form of `M` and `b` in section 3.

- **`2π`-periodicity.** `f` depends on the angles only through `sin θ1`, `sin θ2`, `sin Δ` and `cos Δ`. Hence `f(θ + 2πk, ω) = f(θ, ω)` for integer shifts `k` of either angle. This is why the integrated state never needs wrapping: the vector field cannot tell a wrapped angle from an unwrapped one (D-010).
- **Time reversal.** `b` contains `ω` only through `ω1²` and `ω2²`, so `α(θ, −ω) = α(θ, ω)`: acceleration is **even in `ω`**. With `R(θ, ω) = (θ, −ω)` this gives `R f(R s) = −f(s)`, so if `s(t)` is a solution then `R s(−t)` is one too. This is what makes the "flip the velocities and integrate back" validation meaningful.
- **Reflection.** `M` is even in `θ` and `b` is odd in `θ`, so `α(−θ, ω) = −α(θ, ω)`: acceleration is **odd in `θ`**.

## 6. Equilibria

`f(s) = 0` requires `ω = 0` and `α = 0`. Since `M` is invertible, `α = 0` means `b = 0`, and at `ω = 0` we have `b = −((m1+m2) g L1 sin θ1, m2 g L2 sin θ2)`. Both coefficients are nonzero, so the equilibria are exactly `sin θ1 = sin θ2 = 0`, that is `θi ∈ {0, π}` (mod `2π`): four equilibria, with the hanging state `(0, 0)` and the fully inverted state `(π, π)` among them.

Energies: `V(0, 0) = −[(m1+m2) g L1 + m2 g L2]`. This is the global minimum of `V`, and `T ≥ 0` because `M` is positive definite, so `E` is bounded below by this value at every state, with equality at the hanging state at rest.

**What this cannot catch.** Every equilibrium angle is a multiple of `π`, and `cos(a + b) = cos(a − b)` whenever `sin a sin b = 0`. So an implementation that wrongly uses `cos(θ1 + θ2)` for `cos(θ1 − θ2)` still has all four equilibria and the correct energies there. Only comparisons at generic states against an independent derivation catch that mistake.

## 7. Linearization and normal modes at the hanging state

For small deviations `δ = θ` (with `ω = δ̇`), the velocity terms in `b` are second order, `sin θi ≈ θi`, and `M ≈ M(0)`. Therefore

$$M(0)\,\ddot\delta + K\delta = 0,\qquad K = \partial_\theta^2 V\big|_0 = \mathrm{diag}\bigl((m_1+m_2)gL_1,\ m_2gL_2\bigr).$$

Normal modes `δ ∝ e^{iωt} v` satisfy the generalized eigenproblem `K v = ω² M(0) v`, i.e. `det(K − ω² M(0)) = 0`.

For equal masses and equal lengths (`m`, `L`), `M(0) = m L² [[2, 1], [1, 1]]` and `K = m g L · diag(2, 1)`. With `x = ω² L / g` the condition is `(2 − 2x)(1 − x) − x² = 0`, i.e. `x² − 4x + 2 = 0`, so

$$\omega^2 = \frac{g}{L}\,\bigl(2 \pm \sqrt2\bigr).$$

The Jacobian of `f` at the hanging state has the block form

$$J = \begin{pmatrix} 0 & I \\ -M(0)^{-1}K & 0 \end{pmatrix}.$$

The bottom-right block is zero because `α` is quadratic in `ω`, so its derivative with respect to `ω` vanishes at `ω = 0`. The eigenvalues of `J` are `±iωk`, with `ωk²` the eigenvalues of `M(0)⁻¹K`. These are real and positive. The matrix `M(0)⁻¹K` is **not symmetric** in general, so a symmetric eigensolver must not be used on it.

## 8. Nondimensionalization

Take `L1` as the unit of length, `m1` as the unit of mass, and `√(L1/g)` as the unit of time. With `μ = m2/m1`, `ℓ = L2/L1`, `ω̃ = ω √(L1/g)` and `α̃ = α L1/g`, the equations become

$$\tilde M = \begin{pmatrix} 1+\mu & \mu\ell\cos\Delta \\ \mu\ell\cos\Delta & \mu\ell^2 \end{pmatrix},\qquad
\tilde b = \begin{pmatrix} -\mu\ell\sin\Delta\,\tilde\omega_2^2 - (1+\mu)\sin\theta_1 \\ \mu\ell\sin\Delta\,\tilde\omega_1^2 - \mu\ell\sin\theta_2 \end{pmatrix},\qquad \tilde M\tilde\alpha = \tilde b .$$

So the **dimensionless group is `(μ, ℓ) = (m2/m1, L2/L1)`**, with time measured in units of `√(L1/g)`. The default `m1 = m2 = L1 = L2 = g = 1` is the case `μ = ℓ = 1`; it does not make the other parameters irrelevant, it only fixes the units. Three consequences are tested:

- Scaling **both masses** by `c` leaves `α` unchanged and scales `E` by `c`.
- Scaling `(L1, L2, g)` to `(cL1, cL2, cg)` leaves `g/L` fixed, so `α` is unchanged, and `E` scales by `c²` (both `T ∝ L²` and `V ∝ gL`).
- **Gravity time scaling:** `g → k g` together with `ω → √k ω` multiplies `α` by `k`. This is the statement that time is measured in units of `√(L1/g)`.

Random, distinct, non-default parameters are essential in the tests: with all five parameters equal to one, a swapped or misplaced parameter changes nothing.

## 9. Claims and the tests that check them

Tests live in `tests/`. `O1`–`O7` check the independent oracle itself (SymPy derivation from the geometry of section 2, evaluated at 60 digits); `T1`–`T11` check the owner's code against it. The comparison rule, used everywhere, is `|x − x_ref| ≤ ATOL + RTOL · s`: for value comparisons `s = max(‖x_ref‖∞, 1)`; for residual checks `s` is the sum of the absolute values of the individual terms. Tolerance classes (`IMPL`, `IDENTITY`, `ORACLE_SELF_CHECK`) and their values are in `src/horizon/tolerances.py`, and `check_close` there implements the rule once.

| Claim | Section | Test | Tolerance class |
|---|---|---|---|
| The brief's `T` and `V` equal the Cartesian-geometry ones | 2 | O1 | exact (symbolic) |
| `M` and `b` as written, `det M` in factored form, and the closed form for `α` equal the oracle's | 2, 3 | O3 | exact (symbolic) |
| The oracle's solved accelerations satisfy its Euler-Lagrange equations (a check on the solve step only) | 3 | O2 | `ORACLE_SELF_CHECK` |
| `mass_matrix`, the energies and `acceleration` match the oracle at generic states and at large unwrapped angles | 2, 3, 5 | T1, T2, T3 | `IMPL` |
| The Euler-Lagrange residual of the implemented `acceleration` vanishes (your float64 result is put into the oracle's stored residual and evaluated numerically at 60 digits; nothing is simplified symbolically at test time, see O7) | 3 | T4 | `IMPL`, scale = sum of terms |
| `∇E · f = 0` at every state | 4 | T5 | `IMPL`, scale = sum of terms |
| Acceleration is even in `ω` and odd in `θ`; the vector field is `2π`-periodic in each angle | 5 | T6 | `IDENTITY` |
| The four listed states are equilibria; the hanging energy is `−[(m1+m2) g L1 + m2 g L2]` and is the lower bound of `E` | 6 | T7 | `IDENTITY` |
| `jax.jacfwd` of the vector field equals the oracle's Jacobian (differentiability, needed by the tangent-space method) | 7 | T8 | `IMPL` |
| The oracle's hanging-state linearization has eigenvalues exactly `±i√(2 ± √2)` for the defaults | 7 | O4 | exact (symbolic) |
| The hanging-state Jacobian has blocks `0`, `I`, `0`, and `M(0)⁻¹K` has eigenvalues `2 ± √2` for the defaults, with the 4×4 eigenvalues `±i√(2 ± √2)` | 7 | T9 | `IDENTITY` |
| The dynamics depend only on `m2/m1` and `L2/L1` once time is in units of `√(L1/g)` | 8 | T10 | `IDENTITY` |
| Functions have the documented shapes and `float64`, and agree under `jit` + `vmap` over states and batched parameters passed as traced arguments | interface | T11 | `IDENTITY` |
| The test parameter sets are pairwise distinct by at least 10 percent, and swapping any two of the five parameters is rejected by the `IMPL` tolerance on every set, so a swap cannot hide | 8 | O5 | exact (separation), `IMPL` (swap) |
| The oracle converts floats to high precision exactly | none | O6 | exact |
| The tests of your code run no symbolic computation: the oracle's numeric functions call no SymPy, and the test module reaches the oracle only through them | none | O7 | exact |

**What each family cannot catch.** `T6`, `T7`, `T9` and `T10` are blind to `cos(θ1 + θ2)` in place of `cos(θ1 − θ2)` (section 6). `T5` only compares the energy with the equations, so it also misses a change made consistently to both. Only `T1`–`T4` and `T8`, which compare against the oracle at generic states, catch these mistakes, and they are only as trustworthy as the oracle, which is anchored by `O1`, `O3` and `O4`. `O2` is close to circular (it substitutes the oracle's own solution back in), so it only checks the solve step.
