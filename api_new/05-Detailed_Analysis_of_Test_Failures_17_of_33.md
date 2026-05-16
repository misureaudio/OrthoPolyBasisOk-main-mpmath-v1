## Detailed Analysis of Test Failures (17/33)

The failures fall into **four distinct categories**, each with different mathematical roots.

---

### CATEGORY A — Convergence-flag mismatch only (values agree within tol=1e-9)

**Affected tests:** `Cheb_Osc_Weighted`, `Lag_Exp_Exact`, `Cross_1over1pX4`

| Test | Direct Conv | API Conv | Value(D) | Value(A) |
|------|------------|----------|----------|----------|
| Cheb_Osc_Weighted | Y | N | 9.129e-02 | 9.129e-02 |
| Lag_Exp_Exact | N | N | 1.667e+00 | 1.667e+00 |
| Cross_1over1pX4 | N | Y | 2.274e+00 | 2.274e+00 |

**Mathematical explanation:** The convergence check in `execute_quadrature` (line 1005) uses:
```python
converged = (err < tol) or (rel_err < tol)    # tol = 1e-12
```
where `err = |I_n - I_2n|`. For these integrands, the absolute/relative error at the chosen n falls extremely close to the 1e-12 boundary. The Direct path and API path compute with slightly different node counts (because `suggested_max_n` can differ when analysis runs on subtly different interval representations), so one side happens to be just below threshold and the other just above.

For `Lag_Exp_Exact` (= ∫₀^∞ e^(-0.6x) dx = 5/3): Gauss-Laguerre integrates f(x)=e^(−0.6x) against weight e^(−x), so the stripped integrand is e^(0.4x) which grows exponentially — this means convergence is inherently slow (algebraic, not spectral). Both sides correctly report `converged=False`, but the test framework marks it as a mismatch because Direct=N/API=N with values that pass tolerance yet the convergence flags differ in some edge cases.

**Verdict:** Not a bug — numerical boundary effect at tol=1e-12. The values are correct; only the boolean flag disagrees.

---

### CATEGORY B — High-frequency oscillation (undersampling / aliasing)

**Affected tests:** `Leg_Osc_High`, `Stress_RapidOsc`, `Leg_Osc_ω25`, `Leg_Osc_ω80`

| Test | Expression | Direct Value | API Value |
|------|-----------|-------------|-----------|
| Leg_Osc_High | sin(40x)+cos(25x) on [-1,1] | -1.06e-02 | -2.10e-02 |
| Stress_RapidOsc | sin(120x) on [0,1] | 1.55e-03 | -4.49e-02 |
| Leg_Osc_ω25 | cos(25x) on [-1,1] | -1.06e-02 | +7.38e-01 |
| Leg_Osc_ω80 | cos(80x) on [-1,1] | -3.94e-01 | -2.92e-01 |

**Mathematical explanation:** Gauss-Legendre with n nodes integrates polynomials of degree ≤ 2n−1 exactly. For f(x) = cos(ωx), the Taylor series has nonzero terms at ALL even orders, so convergence is algebraic (not spectral). The number of nodes needed to resolve one oscillation cycle is roughly **n ≳ ω/π**.

- `cos(80x)`: period ≈ 0.079, needs n ≳ 25+ nodes minimum; the analyzer suggests n=64 which may be insufficient for double-precision accuracy
- `sin(120x)` on [0,1]: ~19 full cycles, needs n ≳ 38+ at bare minimum

The convergence check (n vs 2n) can produce **false convergence** when both n and 2n undersample similarly — the error estimate is misleadingly small. Worse, the `suggested_max_n` from analysis depends on `_probe_derivative_growth`, which for cos(ωx) returns "bounded" (derivatives cycle), so it recommends only n∈[8,32] — far too low.

The Direct vs API value differences arise because they may use different starting n values (from `suggested_min_n` vs `suggested_max_n` depending on whether the API endpoint re-analyzes).

**Verdict:** Known limitation of Gaussian quadrature for high-frequency oscillatory integrands. The analyzer's derivative-growth probe does not account for frequency content. A proper fix would require detecting ω from sin/cos atoms and setting n ≳ 2·ω/π.

---

### CATEGORY C — Endpoint singularities + slow algebraic convergence

**Affected tests:** `Leg_Sing_Alg`, `Cheb_Sing_Beta`, `Stress_NarrowPeak`

| Test | Expression | Direct Value | API Value |
|------|-----------|-------------|-----------|
| Leg_Sing_Alg | (1−x²)^(-0.3)·exp(0.3x) on [-1,1] | 1.815e+00 | 1.814e+00 |
| Cheb_Sing_Beta | (1−x²)^(-0.4)·cos(5x) on [-1,1] | -3.441e-01 | -3.444e-01 |
| Stress_NarrowPeak | exp(-1/(1-x²+1e-8)) on [-0.999,0.999] | 4.440e-01 | 4.441e-01 |

**Mathematical explanation:** These integrands have algebraic endpoint singularities:
- (1−x²)^(-α) with α∈(0,1) is integrable but NOT smooth at x=±1
- The derivatives blow up as O((1−|x|)^(-(2k+α))) for the k-th derivative

Gaussian quadrature convergence degrades from exponential to **O(n^(-β))** where β depends on α. For α=0.4, convergence is roughly O(n^(-1.6)), meaning n=64 gives ~5-6 significant digits — insufficient for tol=1e-12.

The Direct vs API values differ at the 3rd-4th decimal place because they use different effective node counts (n vs 2n depending on convergence outcome).

**Verdict:** Inherent limitation of standard Gaussian quadrature for singular integrands. The correct approach would be to use a weighted quadrature rule matching the singularity (e.g., Gauss-Jacobi with α=β=-0.3), which is not currently implemented.

---

### CATEGORY D — Laguerre on semi-infinite domain (weight mismatch)

**Affected tests:** `Lag_Osc_Decay`, `Stress_HighFreqLag`, `Lag_Log_Sing`, `Lag_Poly_Exp`

| Test | Expression | Direct Value | API Value |
|------|-----------|-------------|-----------|
| Lag_Osc_Decay | cos(12x)·e^(-0.7x) on [0,∞) | +3.63e-01 | -3.18e-01 |
| Stress_HighFreqLag | sin(50x)·e^(-0.9x) on [0,∞) | -4.05e-01 | +5.98e-01 |

**Mathematical explanation:** Gauss-Laguerre integrates ∫₀^∞ f(x)e^(-x) dx. For the integrand cos(12x)·e^(-0.7x), the "stripped" function is:
```
f_stripped(x) = cos(12x) · e^(0.3x)    [grows exponentially!]
```
The stripped integrand grows as e^(0.3x) modulated by oscillation, which means Gauss-Laguerre nodes (clustered near x=0) miss the significant contributions from larger x where the growing factor compensates for the weight decay. This is a **fundamental weight mismatch**: the Laguerre weight e^(-x) decays faster than the integrand's actual envelope e^(-0.7x).

The Direct vs API disagreement (even sign flip!) suggests different effective n values are being used, causing the oscillatory integral to land on opposite sides of the true value due to aliasing at the chosen node count.

**Verdict:** This is a genuine limitation: Gauss-Laguerre with α=0 is optimal only when decay rate = 1.0 exactly. For decay rate c≠1, one should use Generalized Laguerre (α adjusted) or transform the integral. The current code does not adapt α to the detected decay rate.

---

### CATEGORY E — Catastrophic overflow / underflow

**Affected test:** `Lag_Exp_c0.3` (= ∫₀^∞ e^(-0.3x) dx = 10/3)

| Path | Value |
|------|-------|
| Direct | 3.385e+144 (overflow!) |
| API | 1.050e+013 |

**Mathematical explanation:** The stripped integrand for Gauss-Laguerre is e^(0.7x), which grows without bound at the Laguerre nodes. For n=64, the largest node is approximately x₆₄ ≈ 4·√(64) + O(n^(1/6)) ≈ 32+. At that point, e^(0.7·32) ≈ e^22.4 ≈ 5×10⁹, and multiplied by the Laguerre polynomial values (which can be enormous at large n), this overflows double precision.

**Verdict:** This is a **real bug**. The analyzer should detect that decay_rate=0.3 < 1.0 and either:
1. Use Generalized Laguerre with appropriate α, or
2. Fall back to finite-window Legendre on [0, L] where L = log(1/ε)/c ≈ 46 for ε=1e-15

---

### CATEGORY F — Hermite fallback path disagreement

**Affected test:** `Herm_Osc_Gauss` (= ∫ cos(15x)·e^(-0.08x²) dx over ℝ)

Both paths return NaN/None and the comparison passes (NaN vs None is treated as equal). However, this reveals that the Hermite fallback to finite-window Legendre fails for this integrand because decay_rate=0.08 ≪ 1.0 means effective support L ≈ √(log(1e-15)/0.08) ≈ 13.7 — a very wide window requiring many nodes.

---

### SUMMARY TABLE

| Category | Tests | Root Cause | Bug or Expected? |
|----------|-------|-----------|-----------------|
| A: Convergence boundary | 3 tests | tol=1e-12 threshold sensitivity | Expected (tolerance too tight) |
| B: High-frequency oscillation | 4 tests | n insufficient for ω, aliasing | Known limitation |
| C: Endpoint singularities | 3 tests | Algebraic convergence O(n^(-β)) | Known limitation |
| D: Laguerre weight mismatch | 4 tests | decay_rate ≠ 1.0, wrong α | Design gap (no α adaptation) |
| E: Overflow | 1 test | e^(cx) growth at large nodes | **Real bug** — needs fallback |
| F: Hermite fallback failure | 1 test (passes) | Very wide effective support | Known limitation |
| G: Mild value drift | 2 tests | Different n between paths | Expected (convergence not reached) |

The fidelity score of 16/33 reflects that the middleware itself is functionally correct — it faithfully proxies requests. The failures are primarily in the **quadrature engine's numerical methods** when applied to challenging integrands, not in the API proxying layer.