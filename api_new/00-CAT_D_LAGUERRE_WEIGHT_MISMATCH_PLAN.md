# CATEGORY D — LAGUERRE WEIGHT MISMATCH IMPROVEMENT PLAN (REVISED)

**Date:** 2026-05-13  
**Source file (READ-ONLY):** `api_new/quadrature_analyzer_laguerre_fixed.py`  
**Output file (NEW COPY, if needed):** `api_new/quadrature_analyzer_d_adapted.py`  

---

## 1. CURRENT STATE AFTER CATEGORY E FIX

The updated test log shows **fidelity improved from 16/33 to 19/33**. The three Laguerre-related gains:

- `Lag_Exp_c0.3`: overflow → correct (CATEGORY E fix)
- `Lag_Exp_Exact` (= exp(-0.6x)): now converged Y/Y, values match ✅
- `Lag_Log_Sing` (= log(1+x)·exp(-0.5x)): now converged Y/Y, values match ✅

### Remaining Laguerre failures (4 tests):

| Test | Expression | Decay c | Stripped g(x) | Direct Conv/API Conv | Values Match? | Root Cause |
|------|-----------|---------|---------------|---------------------|---------------|------------|
| **Lag_Poly_Exp** | x⁵·e^(-1.2x) | 1.2 > 1 | x⁵·e^(−0.2x) [decays] | Y/N | ✅ YES (40.19) | Convergence flag mismatch only |
| **Lag_Osc_Decay** | cos(12x)·e^(-0.7x) | 0.7 < 1 | cos(12x)·e^(+0.3x) [grows] | N/Y | ✅ YES (−0.137) | Convergence flag mismatch only |
| **Stress_HighFreqLag** | sin(50x)·e^(-0.9x) | 0.9 < 1 | sin(50x)·e^(+0.1x) [grows] | N/N | ❌ NO (−0.40 vs +0.60) | **Genuine value mismatch** — undersampling |
| **Lag_Exp_c2.5** | e^(-2.5x) | 2.5 > 1 | e^(+1.5x) [grows] | Y/N | ✅ YES (0.400) | Convergence flag mismatch only |

---

## 2. MATHEMATICAL ANALYSIS OF EACH FAILURE

### 2.1 Lag_Poly_Exp: x⁵·e^(-1.2x), c=1.2 > 1.0

**Stripped integrand:** g(x) = x⁵ · e^(−0.2x) — decays, so standard Gauss-Laguerre is used (no fallback).

The stripped function has a maximum at x = 5/0.2 = 25 and then decays slowly. For n=32 nodes, the largest Laguerre node is ~1024, far beyond where g(x) is significant. The convergence check compares I_32 vs I_64:

- Direct path: converged=True (err < tol at some point in its execution)
- API path: converged=False

**Root cause:** The convergence flag depends on the exact n chosen by `suggested_max_n`, which can differ between paths due to subtle differences in how `_probe_derivative_growth` evaluates x⁵·e^(-1.2x). This is a **tolerance boundary effect**, not a weight mismatch per se.

**Fix needed:** None for accuracy (values match perfectly). The convergence flag inconsistency is a Category A issue, not Category D.

---

### 2.2 Lag_Osc_Decay: cos(12x)·e^(-0.7x), c=0.7 < 1.0

**Stripped integrand:** g(x) = cos(12x) · e^(+0.3x) — grows exponentially, so CATEGORY E fallback triggers → finite-window Legendre on [0, L].

L = log(1e-15)/0.7 ≈ 49.8. The integral is:

```text
∫₀^∞ cos(12x)·e^(-0.7x) dx = 0.7/(0.7² + 144) = 0.7/144.49 ≈ 0.00485
```

The computed value of −0.137 is clearly wrong (should be ~0.005). But both Direct and API agree on this wrong value, so the test passes for fidelity. The convergence flags differ (N vs Y) because one path happens to have err < tol at its particular n while the other doesn't.

**Root cause:** The finite-window Legendre with L=49.8 and n=64 nodes is undersampling a function that oscillates 12/(2π) ≈ 1.9 times per unit over ~50 units — roughly 97 full cycles, but only 64 nodes to resolve them. This produces an inaccurate result, but since both paths produce the same inaccuracy, fidelity passes.

**Fix needed:** Increase n for oscillatory integrands (see Section 3).

---

### 2.3 Stress_HighFreqLag: sin(50x)·e^(-0.9x), c=0.9 < 1.0 — THE CORE CATEGORY D BUG

**Stripped integrand:** g(x) = sin(50x) · e^(+0.1x) — grows (slowly, but still grows), so fallback triggers → finite-window Legendre on [0, L].

L = log(1e-15)/0.9 ≈ 38.4. The integral is:

```
∫₀^∞ sin(ωx)·e^(-cx) dx = ω/(c²+ω²) = 50/(0.81+2500) = 50/2500.81 ≈ 0.01999
```

The test shows Direct=−0.405, API=+0.598 — both wildly wrong AND disagreeing with each other. This is the **genuine value mismatch**.

**Why they disagree:** The convergence check at n vs 2n produces different results depending on which path uses which effective node count:

- Direct might use n=32 → I_32 = −0.405 (undersampled, converged=False)
- API might re-analyze and get a different suggested_max_n → different sampling points

With ω=50, there are ~80 full cycles in [0, 38.4]. With n=64 nodes, that's less than 1 node per cycle — catastrophic undersampling (aliasing). The quadrature result is essentially noise.

**Root cause:** Three compounding issues:

1. **Weight mismatch:** c=0.9 ≠ 1.0 means standard Laguerre weight e^(-x) doesn't match the integrand's decay e^(-0.9x), forcing fallback to finite-window Legendre.
2. **Insufficient n for oscillation frequency:** The analyzer's `_probe_derivative_growth` returns "bounded" for sin(50x)·e^(-0.9x) (derivatives cycle), so it recommends only n∈[8,32] — far too low for ω=50.
3. **No frequency-aware node selection:** The code has no mechanism to detect ω from sin/cos atoms and set n ≳ 10·ω·L/π.

---

### 2.4 Lag_Exp_c2.5: e^(-2.5x), c=2.5 > 1.0

**Stripped integrand:** g(x) = e^(+1.5x) — grows, so CATEGORY E fallback triggers → finite-window Legendre on [0, L].

For c=2.5 > 1.0, the stripped function still grows because we're dividing by e^(-x), not e^(-cx). The growth rate of the stripped function is (c-1) = 1.5, which is substantial. L = log(1e-15)/2.5 ≈ 13.8.

The analytical result: ∫₀^∞ e^(-2.5x) dx = 0.4. The test shows Direct=0.400 (converged=Y), API=0.3999966 (converged=N). Values match within tol, but convergence flags differ.

**Root cause:** Same Category A tolerance boundary effect. Not a weight mismatch issue — the CATEGORY E fallback handles this correctly.

---

## 3. DESIGN PRINCIPLES FOR THE FIX

1. **NEVER modify** `quadrature_analyzer_laguerre_fixed.py` or any module in chebyshev/, hermite/, laguerre/, legendre/
2. Create a new copy: `api_new/quadrature_analyzer_d_adapted.py`
3. The fix targets **only Stress_HighFreqLag** (the genuine value mismatch). The other three failures are convergence-flag mismatches with correct values — Category A, not Category D.

---

## 4. GENERAL FREQUENCY DETECTION: WHY THE SIMPLE APPROACH IS INSUFFICIENT

### 4.1 The Problem with `as_coeff_mul(v)`

The original plan used `arg.as_coeff_mul(v)[0]` to extract ω from sin(ωx). This works for **linear** arguments but fails for composite functions:

| Expression | Phase φ(x) | Instantaneous ω = |φ'(x)| | Simple approach result |
|-----------|-----------|-------------------|----------------------|
| sin(50x) | 50x | 50 (constant) | ✅ 50 |
| sin(exp(x)) | exp(x) | exp(x) (grows!) | ❌ Returns 1 (coefficient of x in exp(x) is not meaningful) |
| sin(x²) | x² | 2x (linear growth) | ❌ Returns 1 |
| sin(3·exp(0.5x)) | 3·exp(0.5x) | 1.5·exp(0.5x) | ❌ Returns 1 |

For `sin(exp(x))` on [0, 10]: ω_max = exp(10) ≈ 22026 — the function oscillates ~35,000 times! The simple approach would miss this entirely and use n=64 nodes.

### 4.2 General Approach: Instantaneous Frequency via Phase Derivative

For any f(x) = sin(φ(x)) or cos(φ(x)):
- **Instantaneous angular frequency:** ω_inst(x) = |dφ/dx|
- **Maximum on [a, b]:** ω_max = max_{x∈[a,b]} |φ'(x)|

This is exact and handles all cases:

```python
def _estimate_max_oscillation_frequency(self, expr, variable: str,
                                          a: float, b: float) -> float:
    """Estimate the maximum instantaneous angular frequency of oscillatory components.

    For f(x) = sin(φ(x)) or cos(φ(x)): ω_inst(x) = |dφ/dx|.
    Returns max_{x∈[a,b]} |φ'(x)| over all trig atoms in expr.

    Handles:
      - Linear: sin(ωx) → ω (constant)
      - Quadratic: sin(x²) → 2b
      - Exponential: sin(exp(x)) → exp(b)
      - Composite: sin(3·exp(0.5x)) → 1.5·exp(b)

    Returns 0.0 if no oscillatory components detected.
    """
    v = Symbol(variable)
    max_freq = 0.0

    trig_atoms = list(expr.atoms(sin)) + list(expr.atoms(cos))
    for sub in trig_atoms:
        try:
            phase = sub.args[0]          # φ(x)
            dphase = diff(phase, v)      # φ'(x)

            # Strategy 1: If |φ'| is monotonic on [a,b], evaluate at endpoints
            # This covers most practical cases (linear, quadratic, exponential phases)
            val_a = abs(float(N(dphase.subs(v, a), 15)))
            val_b = abs(float(N(dphase.subs(v, b), 15)))

            if math.isfinite(val_a): max_freq = max(max_freq, val_a)
            if math.isfinite(val_b): max_freq = max(max_freq, val_b)

            # Strategy 2: If |φ'| has an interior extremum, probe at midpoint too
            mid = (a + b) / 2.0
            val_mid = abs(float(N(dphase.subs(v, mid), 15)))
            if math.isfinite(val_mid): max_freq = max(max_freq, val_mid)

        except Exception:
            pass

    return max_freq
```

### 4.3 Why This Works for All Cases

| Expression | φ'(x) | Evaluated at a=0, b=L | Result |
|-----------|-------|----------------------|--------|
| sin(50x) | 50 | 50, 50 | ✅ ω_max = 50 |
| sin(exp(x)) on [0,10] | exp(x) | 1, 22026 | ✅ ω_max = 22026 |
| sin(x²) on [0,10] | 2x | 0, 20 | ✅ ω_max = 20 |
| sin(3·exp(0.5x)) on [0,10] | 1.5·exp(0.5x) | 1.5, 64.9 | ✅ ω_max = 64.9 |

The three-point probe (a, midpoint, b) catches monotonic and unimodal frequency profiles. For pathological cases with multiple interior extrema of φ'(x), the midpoint may miss them — but this is acceptable because:
1. Such functions are extremely rare in practice
2. The convergence check (n vs 2n comparison) will detect insufficient resolution
3. We can always increase the safety factor

---

## 5. DETAILED FIX PLAN

### Step 1: Create the copy

```bash
copy api_new\quadrature_analyzer_laguerre_fixed.py api_new\quadrature_analyzer_d_adapted.py
```

### Step 2: Add general frequency estimation helper (NEW)

Replace the simple `_extract_oscillation_frequencies` with the robust `_estimate_max_oscillation_frequency`:

```python
def _estimate_max_oscillation_frequency(self, expr, variable: str,
                                          a: float, b: float) -> float:
    """Estimate the maximum instantaneous angular frequency of oscillatory components.

    For f(x) = sin(φ(x)) or cos(φ(x)): ω_inst(x) = |dφ/dx|.
    Returns max_{x∈[a,b]} |φ'(x)| over all trig atoms in expr.

    Handles:
      - Linear: sin(ωx) → ω (constant)
      - Quadratic: sin(x²) → 2b
      - Exponential: sin(exp(x)) → exp(b)
      - Composite: sin(3·exp(0.5x)) → 1.5·exp(b)

    Returns 0.0 if no oscillatory components detected.
    """
    v = Symbol(variable)
    max_freq = 0.0

    trig_atoms = list(expr.atoms(sin)) + list(expr.atoms(cos))
    for sub in trig_atoms:
        try:
            phase = sub.args[0]          # φ(x)
            dphase = diff(phase, v)      # φ'(x)

            # Three-point probe: endpoints and midpoint
            for pt in (a, (a + b) / 2.0, b):
                try:
                    val = abs(float(N(dphase.subs(v, pt), 15)))
                    if math.isfinite(val):
                        max_freq = max(max_freq, val)
                except Exception:
                    pass

        except Exception:
            pass

    return max_freq
```

### Step 3: Add frequency-aware n adjustment (NEW)

Replace `_compute_oscillation_safe_n` with a version that uses the general estimator:

```python
def _compute_oscillation_safe_n(self, expr, variable: str,
                                  interval_a: float, interval_b: float,
                                  base_n: int) -> int:
    """Ensure enough nodes per oscillation cycle.

    For sin(φ(x)) or cos(φ(x)): need at least ~10 nodes per half-period.
    Total cycles in [a,b]: ∫_a^b |φ'(x)|/(2π) dx ≈ (b-a)·ω_max/(2π).
    Nodes needed: 10 · ω_max · (b-a) / π.

    Returns max(base_n, min_osc_nodes), capped at 500 to prevent runaway.
    """
    omega_max = self._estimate_max_oscillation_frequency(
        expr, variable, interval_a, interval_b
    )
    if omega_max < 1e-6:
        return base_n

    # Number of half-periods in [a, b] (worst case: constant ω = ω_max)
    n_half_periods = omega_max * (interval_b - interval_a) / math.pi
    # Need at least 10 nodes per half-period for spectral accuracy
    min_osc_nodes = int(10 * n_half_periods)

    return max(base_n, min(min_osc_nodes, 500))
```

### Step 4: Modify `_integrate_laguerre` fallback path

In the finite-window Legendre fallback (triggered when stripped integrand grows), increase the node count based on oscillation frequency:

**Current code:**
```python
return self._integrate_legendre(func_original, 0.0, L, max(n * 2, 64))
```

**New code:**
```python
osc_safe_n = self._compute_oscillation_safe_n(
    expr, variable, 0.0, L, max(n * 2, 64)
)
return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)
```

### Step 5: Modify `_integrate_hermite` fallback path (same pattern)

The Hermite fallback at lines 1127-1131 has the same issue for oscillatory integrands on infinite domains. Apply the same fix:

**Current code:**
```python
return self._integrate_legendre(func_original, -L, L, max(n * 2, 64))
```

**New code:**
```python
osc_safe_n = self._compute_oscillation_safe_n(
    expr, variable, -L, L, max(n * 2, 64)
)
return self._integrate_legendre(func_original, -L, L, osc_safe_n)
```

---

## 6. EXPECTED IMPROVEMENTS

| Test | Before Fix | After Fix (expected) | Analytical Value |
|------|-----------|---------------------|-----------------|
| Stress_HighFreqLag | Direct=−0.40, API=+0.60 | Both ≈ 0.020 | 50/2500.81 ≈ 0.020 |
| Lag_Osc_Decay | −0.137 (both agree but wrong) | ≈ 0.0049 | 0.7/144.49 ≈ 0.0049 |

**Node count estimates with the fix:**

| Test | ω_max | L | Half-periods | Nodes needed (capped at 500) |
|------|-------|-----|-------------|------------------------------|
| Stress_HighFreqLag: sin(50x)·e^(-0.9x) | 50 | 38.4 | ~796 | **500** (capped) |
| Lag_Osc_Decay: cos(12x)·e^(-0.7x) | 12 | 49.8 | ~191 | **~1910 → capped at 500** |

Note: Even with the fix, these values may not reach machine precision because finite-window truncation introduces tail error and n=500 is still modest for such high-frequency integrals. But they should be within tol=1e-9 of each other (fidelity pass) and much closer to analytical values than before.

---

## 7. WHAT THIS FIX DOES NOT ADDRESS

The following failures are **NOT Category D** and require separate plans:

| Test | Category | Reason |
|------|----------|--------|
| Lag_Poly_Exp (conv flag only) | A — Convergence boundary | Values match; flag differs at tol=1e-12 threshold |
| Lag_Osc_Decay (conv flag only) | A + B — Boundary + undersampling | Values match between paths; both are inaccurate but that's a separate issue |
| Lag_Exp_c2.5 (conv flag only) | A — Convergence boundary | Values match; flag differs at tol=1e-12 threshold |

---

## 8. FILES TO CREATE/MODIFY

| Action | File |
|--------|------|
| COPY → MODIFY | `api_new/quadrature_analyzer_laguerre_fixed.py` → `api_new/quadrature_analyzer_d_adapted.py` |
| NO CHANGE | All other files (including laguerre/, legendre/ modules) |