## CATEGORY D Improvement — Complete

**New file:** `api_new/quadrature_analyzer_d_adapted.py` (based on `quadrature_analyzer_laguerre_fixed.py`)

### Changes made:

1. **`_estimate_max_oscillation_frequency(expr, var, a, b)`** — General frequency detection using phase derivative φ'(x) evaluated at {a, midpoint, b}. Handles linear sin(ωx), quadratic sin(x²), exponential sin(exp(x)), and composite phases.

2. **`_compute_oscillation_safe_n(expr, var, interval_a, interval_b, base_n)`** — Computes minimum node count: `max(base_n, min(10·ω_max·Δx/π, 500))`.

3. **Modified `_integrate_laguerre()` with three-layer defense:**
   - Layer 1 (CATEGORY E): overflow detection at probe points → Legendre fallback
   - Layer 2 (CATEGORY E): exponential growth ratio > 1e6 → Legendre fallback  
   - Layer 3 (**NEW CATEGORY D**): oscillations detected + any stripped growth (>1.5x) → Legendre fallback with oscillation-aware n

4. **Modified `_integrate_hermite()` fallback path** to use oscillation-aware node count.

### Test results (all Laguerre cases):

| Expression | Before Fix | After Fix | Expected | Status |
|-----------|-----------|-----------|----------|--------|
| exp(-0.3x) | 3.385e+144 ❌ | 3.333 ✅ | 3.333 | CATEGORY E fix |
| exp(-0.6x) | 1.667 ✅ | 1.667 ✅ | 1.667 | No regression |
| x⁵·exp(-1.2x) | 40.19 ✅ | 40.19 ✅ | — | No regression |
| cos(12x)·e^(-0.7x) | −0.137 ❌ | 0.00485 ✅ | 0.00485 | **CATEGORY D fix** |
| log(1+x)·e^(-0.5x) | 1.846 ✅ | 1.846 ✅ | — | No regression |
| exp(-x) | 1.000 ✅ | 1.000 ✅ | 1.000 | No regression |
| sin(50x)·e^(-0.9x) | −0.405 ❌ | 0.020 ✅ | 0.020 | **CATEGORY D fix** |
| exp(-1.1x) | 0.909 ✅ | 0.909 ✅ | 0.909 | No regression |
| exp(-2.5x) | 0.400 ✅ | 0.400 ✅ | 0.400 | No regression |

### Files created/modified:
- `api_new/quadrature_analyzer_laguerre_fixed.py` — CATEGORY E fix (overflow protection)
- `api_new/quadrature_analyzer_d_adapted.py` — CATEGORY D fix (oscillation awareness, builds on E)
- All original files untouched (`quadrature_analyzer_hermite_improved.py`, laguerre/, legendre/ modules)