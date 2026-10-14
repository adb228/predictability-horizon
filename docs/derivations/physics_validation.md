# Physics validation: energy drift, the small-angle FFT and time reversibility

This note says what the milestone 4 tests claim, why the claims are true or approximately true, and what each test cannot see. It covers the brief's validation tests 1 to 3 (section 3 of `docs/BRIEF.md`). Tests 4 and 5, convergence order and agreement of RK4 with DOP853, were delivered in milestone 3 (G1 to G4, `docs/derivations/integrators.md`). Every number in the tests lives in `src/horizon/tolerances.py` and was logged in D-052 to D-060 before any test ran. The measurements the gates were sized from are uncommitted lab work and are quoted only in the decision log, labelled as such.

## 1. Conventions and units

State `(θ1, θ2, ω1, ω2)`, parameters `(m1, m2, L1, L2, g)`, angles never wrapped. Time is in whatever unit the vector field uses; for the pendulum that is `t_s = √(L1/g)`, which is 1 for the default parameters (`docs/derivations/double_pendulum.md`, section 8). The gates that use the default parameters state times in that unit. The second small-angle set (F2 below) has `t_s ≠ 1`, and its sampling and step are stated in units of its own `t_s`, while its frequencies are compared in physical units.

`S` flips the velocities: `S(θ1, θ2, ω1, ω2) = (θ1, θ2, −ω1, −ω2)`.

Two integrators are tested: DOP853 through `solve_reference` at its defaults (`rtol = atol = 1e-12`, the brief's) and fixed-step RK4 through `rk4_trajectory` at `dt = 0.01` (`AGREE_DT`, the step of milestone 3).

Test families: **E** energy drift, **F** small-angle FFT, **V** reversibility, **N** negative controls, **C** seeded-case recipes, **K** the register and the helpers.

## 2. Energy

### 2.1 The statistic, and why the brief's wording is an error

`E` is `double_pendulum.energy` (milestone 2). Along the exact flow `E` is constant (T5 of milestone 2). For a computed trajectory the drift is

`D = max over the output times t of |E(t) − E(0)| / E_s`, with `E_s = g ((m1 + m2) L1 + m2 L2)`.

`E_s` is the largest possible `|V|`. It is a fixed scale set by the parameters alone, not by the state or the trajectory, so one number applies to every state of a parameter set; the code never uses a state-dependent form such as `max(|E(0)|, E_s)` or `max |E(t)|`. Shifting `E` by a constant leaves `E(t) − E(0)` and `E_s` unchanged, so `D` does not depend on where the potential energy is zero.

The brief's test 1 says "relative energy drift ... below ~1e-9". Read literally, that is `|E(t) − E(0)| / |E(0)|`, and the reading is an error in the brief, logged as one in D-053 (the brief itself is not edited):

- it is undefined at `E(0) = 0` and ill-conditioned near it;
- `|E(0)|` depends on the arbitrary zero of the potential energy, so a fixed threshold on the ratio is not a well-posed statement about the integrator;
- `|E(t) − E(0)|` is almost independent of the state, so the ratio mostly measures how large the initial energy happens to be.

What is wrong is the normaliser. The number `1e-9` is kept as the E1 ceiling. As a transparency measure the script also prints the literal ratio, non-gating, for the states whose `|E(0)|` is at least `0.1 E_s`; states below that floor are counted and their `E(0)` listed, with no ratio printed. The literal ratio can still exceed `1e-9` for a state above the floor (it does whenever `|E(0)| < D E_s / 1e-9`), and the report says so where it happens. The floor test uses `E` as `double_pendulum.energy` defines it, which fixes the zero of the potential, and `E_s` from `energy_scale`.

### 2.2 The ensembles

Three kinds of initial state, all at rest unless stated:

- **regular**: both angles uniform in `[−0.4, 0.4]` (near the linear regime, `E(0)` close to the hanging value `−E_s`);
- **chaotic-ish**: both angles uniform in `[−π, π]`, kept only if `E > 0`, which at rest means `V > 0` (the recipe of D-046(a); not a classification, milestone 6 decides that);
- **moving**: angles uniform in `[−π, π]` and velocities uniform in `[−2, 2]`. These are measured and reported, not gated, because their drift comes within about 1.1 to 1.3 times of `1e-9` at the brief's tolerance (D-054).

### 2.3 Why integrators drift, and what E1 and E2 test

The exact flow conserves `E`. A one-step method makes a small error at every step, and nothing forces the energy errors of successive steps to cancel unless the method is built to conserve energy (a symplectic method applied in canonical coordinates keeps the energy error bounded over very long times; neither RK4 nor DOP853 is symplectic). So the drift grows with the number of steps. How it grows is measured, not assumed: the script prints the running maximum of `D` at `t = 100`, `300` and `1000` for every gate state.

- **RK4** has local error `O(dt⁵)`, hence an energy drift of order `dt⁴` at a fixed `T` (section 2.3 of the integrators note); halving `dt` should reduce it by about 16. **E2** gates it at `dt = 0.01` with ceilings per ensemble, so that a gross error (a wrong stage, a wrong step scaling) shows at a long horizon. The order itself is gated by G1 and G2.
- **DOP853** controls the local error of each step to `rtol = atol = 1e-12` (section 4 of the integrators note). That is not a bound on the global error, so the drift depends on the tolerance and on the number of steps. **E1** is the brief's test: `D ≤ 1e-9` at `T = 1000`. The control with `rtol = atol = 1e-10` shows that E1 can tell a looser reference from the default.

E1 and E2 are blind to an error that conserves `E`, for example a wrong time scaling `g/L` applied consistently to the field and to `E`. Those are the job of the oracle comparisons of milestone 2, of F, and of the wiring check of milestone 3.

## 3. The small-angle FFT

### 3.1 The linearisation

Near the hanging state, with small displacement `δ`, `M(0) δ'' + K δ = 0` with `K = diag((m1 + m2) g L1, m2 g L2)` (section 7 of the milestone 2 note). Normal modes `δ ∝ e^{iωt} v` satisfy `K v = ω² M(0) v`. For equal masses and lengths, with `x = ω² L / g`, the condition is `x² − 4x + 2 = 0`, so

`ω² = (g/L)(2 ± √2)`, `ω1 = √(2 − √2) √(g/L) ≈ 0.7654 √(g/L)`, `ω2 = √(2 + √2) √(g/L) ≈ 1.8478 √(g/L)`.

The first row of `(K − ω² M(0)) v = 0` gives `(2 − 2x) v1 = x v2`. For `x = 2 − √2`, `v2/v1 = (2 − 2x)/x = √2`, the slow in-phase mode; for `x = 2 + √2`, `v2/v1 = −√2`, the fast out-of-phase mode. The tests start from `(A, √2 A)`, `(A, −√2 A)` (one mode each) and `(A, 0)` (both modes with equal weight in `θ1`), all with `ω = 0`.

### 3.2 The signal, and a physical error that is not the FFT's

`θ1(t)` is a sum of oscillations at `ω1` and `ω2` plus nonlinear corrections of higher order in `A`. A pendulum's frequency depends on its amplitude, `ω(A) = ω0 (1 + c A² + …)`, so the peak of a finite-amplitude simulation sits slightly away from the linear frequency. That shift is physics and not numerical error, and it sets a floor under the achievable agreement: it scales as `A²`, which is why the amplitude `A = 1e-3` is part of the declared values and is not adjusted to fit (D-056 gives the measured coefficients).

### 3.3 The estimator

The record has `N = 20000` samples at spacing `h = 0.05 t_s`, so `T_total = N h = 1000 t_s`. The resolution of the raw spectrum is `2π / T_total` in `ω`, far coarser than the accuracy wanted, so the peak is located by three steps:

1. remove the mean and multiply by a Hann window (`np.hanning`), which suppresses the leakage of a finite record at the price of a wider main lobe (4 original bins);
2. zero-pad to a power of two of at least `8 N` (here `2^18 = 262144`, a padding factor of about 13), which samples the spectrum more densely and does not add information;
3. take the `n` largest local maxima of `|X_k|` that are more than 40 padded bins apart (the main lobe is about 52 padded bins wide here; the separation only has to exclude a second maximum on the same peak, and the two real peaks are over 2000 bins apart), and refine each by a parabola through the logarithms of the magnitudes of the peak bin and its two neighbours: with `a, b, c = ln|X_{k−1}|, ln|X_k|, ln|X_{k+1}|`, the offset is `δ = ½ (a − c) / (a − 2b + c)` in bins.

`n = 1` for the single-mode states and `n = 2` for the generic state. The analytic frequencies are used only to compare, never to find the peaks. The analytic values are written in the test from `2 ± √2` and `g/L`, not taken from the code under test.

### 3.4 The two parameter sets

**F1** is the default set. **F2** is `m1 = m2 = 1.3`, `L1 = L2 = 0.7`, `g = 9.81`. The physics of F2 is the physics of F1 in rescaled time, so a correct code gives the same relative errors; F2 therefore tests the `g/L` scaling of the dynamics, the time unit, and the way the sampling is stated in `t_s`, and nothing else. A fully general parameter set is not tested here, because T8 and T9 of milestone 2 already check the general linearisation against the oracle.

F is blind to a swap of two parameters that are equal in the set (the default set has all five equal), to an error below the tolerance, and to a defect that changes no frequency.

## 4. Time reversibility

### 4.1 The symmetry of the equations

By T6 of milestone 2 the acceleration is even in `ω`: `α(θ, −ω) = α(θ, ω)`. Let `y(t) = (θ(t), ω(t))` solve `θ' = ω`, `ω' = α(θ, ω)` and set `z(t) = S y(−t) = (θ(−t), −ω(−t))`. Then `θ_z' = −ω(−t) = ω_z`, and `ω_z' = ω'(−t) = α(θ(−t), ω(−t)) = α(θ_z, −ω_z) = α(θ_z, ω_z)`, so `z` solves the same equations. Consequence: from `y0`, integrate forward for `T` to reach `y_T`, flip to `S y_T`, integrate forward for `T`, and flip back: the result is `y0`. This is the protocol of the brief's test 3, and for the exact flow it holds exactly.

### 4.2 What a numerical method does

A numerical method `Φ_h` only approximates the flow, and nothing forces the round trip to cancel the errors of the two legs: they add, and in a chaotic region they are amplified. For a Runge–Kutta method applied to a field with `f(S y) = −S f(y)` (which is the same statement as T6, together with the fact that `θ' = ω` flips sign), `Φ_{−h}(y) = S Φ_h(S y)` holds exactly, because each stage of the step with `+h` from `S y` equals `−S` times the matching stage of the step with `−h` from `y` (by induction on the stages: `f(S(y + c k)) = −S f(y + c k)`), and the update is linear in the stages. So for RK4 a negative step and the flip-integrate-flip protocol are the same operation; test **V3** checks that identity and it is not a second reversibility gate. It also checks the statement of D-043 that a negative `dt` is legal and is what milestone 5 and the reversibility tests need.

### 4.3 Chaos limits the horizon

An error `ε` introduced early in a chaotic region grows like `ε e^{λt}` over the rest of the run, so the error of a round trip grows exponentially with `T`, and the round trip is only meaningful for a short `T`. In a regular, near-linear regime errors grow slowly, so a long `T` is fine. The tests therefore use `T = 100` for regular states and `T = 8` for chaotic-ish states (the horizon of milestone 3's agreement gate); `T` is a design variable and will be restated in Lyapunov times after milestone 5.

V is blind to an error that is itself time-reversal symmetric, because such an error cancels in the round trip: a wrong value of `g`, a wrong mass ratio, a wrong time scaling. Those are the job of the oracle comparisons, of F and of the wiring check.

## 5. Declared values (declared, not measured)

All values were logged in D-052 to D-060 before any milestone 4 test ran, and live in `src/horizon/tolerances.py`, pinned by the tripwire test K1. Changing any of them needs Opus 5.5's approval first (D-025). The streams are `np.random.default_rng([CASES_SEED, stream])`.

| Gate | Declared value | Decision |
|---|---|---|
| energy statistic | `max_t |E(t) − E(0)| / E_s`, `E_s` from the parameters alone | D-053 |
| E1 | DOP853 defaults, `T = 1000`, integer times; ceiling `1e-9` (the brief's number); 2 regular states (stream 3000) and 5 chaotic-ish states (stream 3001); each test has `@pytest.mark.timeout(300)` | D-054 |
| E1 reports | running maximum at `t = 100`, `300`, `1000`; moving states (stream 3006, 8 states, `ω` in `[−2, 2]`) not gated; literal ratio shown where `|E(0)| ≥ 0.1 E_s` | D-053, D-054 |
| E1 controls | damping `γ = 1e-6`; `rtol = atol = 1e-10` on the first two chaotic-ish states | D-054, D-058 |
| E2 | RK4 `dt = 0.01`, `T = 1000`; 12 regular states (stream 3002), ceiling `1e-7`; 30 chaotic-ish states (stream 3003), ceiling `5e-4` | D-055 |
| E2 control | RK4 at `dt = 0.05` | D-058 |
| F | `A = 1e-3`, `h = 0.05 t_s`, 20000 samples, RK4 `dt = 0.01 t_s` with stride 5, Hann, padding at least 8, peak separation 40 padded bins, relative error `≤ 1e-5`; sets F1 (default) and F2 (`1.3, 1.3, 0.7, 0.7, 9.81`) | D-056 |
| F controls | `m2` scaled by `1.01`; the pair `2 ± 1` | D-058 |
| V1, V2 | forward `T`, flip, forward `T`, flip back; regular `T = 100` (stream 3004, 12 states), chaotic-ish `T = 8` (stream 3005, 30 states); DOP853 ceilings `5e-9` and `5e-8`, RK4 ceilings `1e-6` and `2e-3`; the V1 test on regular states has `@pytest.mark.timeout(120)` | D-057, D-063 |
| V controls | damped field (V1); wrong schemes W1, W2, W3 of milestone 3 on regular states and W3 on chaotic-ish states (V2) | D-058 |

The rule for every calibrated ceiling (D-055): 10 times the largest of 20 disjoint calibration sample maxima, rounded up to 1, 2 or 5 times a power of ten. Two values are not set by it: the E1 ceiling `1e-9`, which the brief fixes, and the F tolerance `1e-5`, whose cases are deterministic.

## 6. Claims and the tests that check them

Tests are in `tests/test_physics_validation.py` (E, F, V, N) and `tests/test_physics_register.py` (C, K); the shared definitions and measurements are in `tests/physics_checks.py`, which the script `experiments/stage1/physics_validation.py` also uses, so a gate and the number printed for it have one implementation.

| Claim | Section | Test | Rule |
|---|---|---|---|
| DOP853 at its defaults keeps `D ≤ 1e-9` up to `t = 1000` on the 2 regular and 5 chaotic-ish gate states | 2.3 | E1 | ceiling (the brief's number) |
| A damping term `γ = 1e-6` in the field, and a reference tolerance of `1e-10`, both fail E1 | 2.3 | E1 controls | worst case `>` ceiling |
| RK4 at `dt = 0.01` keeps `D` under the ceilings of D-055 on the regular and chaotic-ish ensembles | 2.3 | E2 | ceiling |
| RK4 at `dt = 0.05` fails E2 | 2.3 | E2 control | worst case `>` ceiling |
| The FFT peaks of `θ1(t)` equal `√((g/L)(2 ± √2))` to `1e-5`, from each mode and from the generic state, for both integrators and both sets | 3 | F1, F2 | relative error `≤` tolerance |
| `m2` scaled by `1.01`, or the pair `2 ± 1`, fails the FFT test | 3 | F controls | error `>` tolerance |
| The round trip returns to the initial state within the ceilings of D-057, for DOP853 and RK4, on regular and chaotic-ish states | 4 | V1, V2 | ceiling |
| RK4's negative-`dt` trajectory equals the flipped forward trajectory | 4.2 | V3 | `IDENTITY` |
| A damped field fails V1; W1, W2, W3 fail V2 on regular states and W3 on chaotic-ish states | 4 | V controls | error `>` ceiling |
| The seeded cases follow the logged recipes and are reproducible | 2.2, 3.1, 4 | C1 to C3 | exact |
| The register holds the declared values, and the energy statistic takes its scale from the parameters alone | 2.1, 5 | K1 | exact |
| The helpers behave: the peak finder recovers known frequencies, the statistic ignores the zero of the potential, the literal-ratio floor excludes the right states | 2.1, 3.3 | K2 | exact |

**What each family cannot catch.**

- **E1 and E2** are blind to energy-conserving errors, and E1 is blind to moving states (reported, not gated). E2 on chaotic-ish states cannot tell a method of order 4 from one of order 3 at `dt = 0.01`; G1 and G2 do that.
- **F** is blind to swapped equal parameters, to errors below `1e-5`, and to defects that change no frequency.
- **V** is blind to time-reversal-symmetric errors. On chaotic-ish states V2 is a backstop for gross errors: second-order schemes such as W1 and W2 sit near its ceiling at `dt = 0.01` (their round-trip error is between 0.4 and 2.7 times the ceiling, depending on the sample: below it on some samples and above it on the gate ensemble), so V2 may or may not see them; G1 and G2 are what catch them (D-063).
- **Not covered here:** the Lyapunov estimators (milestone 5), energy regimes and finite-time exponents (milestone 6), and the report (milestone 7).
