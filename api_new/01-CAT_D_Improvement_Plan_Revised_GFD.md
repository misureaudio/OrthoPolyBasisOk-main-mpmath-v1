## CATEGORY D Improvement Plan — Revised with General Frequency Detection

**Document:** `CATEGORY_D_LAGUERRE_WEIGHT_MISMATCH_PLAN.md` (updated)

### Answering the question: Is simple frequency detection sufficient?

**No.** The original approach using `arg.as_coeff_mul(v)[0]` only handles linear arguments like sin(ωx). It fails catastrophically for composite functions:

| Expression | Phase φ(x) | Instantaneous ω = |φ'(x)| | Simple approach |
|-----------|-----------|-------------------|-----------------|
| sin(50x) | 50x | 50 (constant) | ✅ Returns 50 |
| sin(exp(x)) on [0,10] | exp(x) | exp(x), max=22026! | ❌ Returns 1 |
| sin(x²) on [0,10] | x² | 2x, max=20 | ❌ Returns 1 |

For `sin(exp(x))` on [0,10], the function oscillates ~35,000 times — but the simple approach would use n=64 nodes.

### The general solution: Phase derivative evaluation

Instead of extracting ω from coefficients, compute φ'(x) symbolically and evaluate at probe points {a, midpoint, b}:

```
ω_max = max_{pt ∈ {a, (a+b)/2, b}} |φ'(pt)|
```

This is **exact** for all cases:
- sin(50x): φ'=50 → ω_max=50 ✅
- sin(exp(x)) on [0,10]: φ'=exp(x) → ω_max=exp(10)=22026 ✅  
- sin(x²) on [0,10]: φ'=2x → ω_max=20 ✅

### The revised plan (Section 5):
1. **`_estimate_max_oscillation_frequency(expr, var, a, b)`** — general phase derivative approach
2. **`_compute_oscillation_safe_n(expr, var, interval_a, interval_b, base_n)`** — nodes = max(base_n, min(10·ω_max·Δx/π, 500))
3. Apply to both Laguerre and Hermite fallback paths

### Limitations acknowledged:
- Three-point probe catches monotonic/unimodal frequency profiles (covers all practical cases)
- Pathological φ'(x) with multiple interior extrema may be missed — but the convergence check (n vs 2n) will detect this
- Cap at n=500 prevents runaway for extreme frequencies like sin(exp(x)) on [0,10]