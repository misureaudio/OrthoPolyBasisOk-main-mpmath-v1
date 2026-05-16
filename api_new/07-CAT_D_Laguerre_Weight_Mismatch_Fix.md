## CATEGORY D Laguerre Weight Mismatch Fix — Complete

**New file:** `api_new/quadrature_analyzer_d_adapted.py` (copy of `quadrature_analyzer_laguerre_fixed.py` with CATEGORY D additions)

### Changes applied:

1. **`_estimate_max_oscillation_frequency(expr, var, a, b)`** — General frequency detection using phase derivative φ'(x). Evaluates at {a, midpoint, b} to find max ω on the interval. Handles linear (sin(ωx)), quadratic (sin(x²)), exponential (sin(exp(x))), and composite phases.

2. **`_compute_oscillation_safe_n(expr, var, a, b, base_n)`** — Computes minimum node count: `max(base_n, min(10·ω_max·Δx/π, 500))`. Capped at 500 to prevent runaway for extreme frequencies.

3. **Modified `_integrate_laguerre` standard path** — When oscillations are detected AND the stripped function shows any growth (even mild, ratio > 1.5), falls back to finite-window Legendre on [0, L] with oscillation-aware node count instead of increasing Laguerre nodes (which would overflow at large x).

4. **Modified `_integrate_hermite` fallback path** — Uses oscillation-aware node count in the Legendre fallback.

### Test results:

| Expression | Before Fix | After Fix | Analytical Value | Status |
|-----------|-----------|-----------|-----------------|--------|
| exp(-0.3x) | 3.385e+144 (overflow!) | 3.333... | 3.333... | ✅ CATEGORY E fix |
| exp(-0.6x) | slow convergence | 1.667 | 1.667 | ✅ CATEGORY E fix |
| cos(12x)·e^(-0.7x) | −0.137 (wrong!) | 4.845e-3 | 4.845e-3 | ✅ FIXED by D |
| sin(50x)·e^(-0.9x) | −0.40 vs +0.60 (wrong!) | 1.999e-2 | 1.999e-2 | ✅ FIXED by D |
| exp(-x) | crashed (TypeError) | 1.000 | 1.000 | ✅ CATEGORY E fix |
| log(1+x)·e^(-0.5x) | slow convergence | 1.846 | — | ✅ CATEGORY E fix |
| exp(-1.1x) | correct | 0.909 | 0.909 | ✅ No regression |
| exp(-2.5x) | correct | 0.400 | 0.400 | ✅ No regression |

### Files created (no originals modified):
- `api_new/quadrature_analyzer_laguerre_fixed.py` — CATEGORY E fix only
- `api_new/quadrature_analyzer_d_adapted.py` — CATEGORY E + D fixes combined