# CATEGORY C: Endpoint Singularities with Slow Algebraic Convergence — Analysis & Plan

## Problem Statement

Certain test cases have **integrable endpoint singularities** where the function value diverges algebraically (f(x) ~ |x - x₀|^(-α), 0 < α < 1). Standard Gauss-Legendre quadrature converges with only **algebraic rate O(n^(-p))** for these functions, rather than exponential convergence. This means they never reach tol=1e-12 within practical node counts.

### Affected Test Cases (all show N/N — neither Direct nor API converges)

| Test | Expression | Singularity Type | Convergence Rate |
|------|-----------|------------------|-----------------|
| Leg_Sing_Alg | (1-x²)^(-0.3)·exp(0.3x) on [-1,1] | Algebraic blowup at x=±1 | Very slow: n=256→error ~1e-3 |
| Cheb_Sing_Beta | (1-x²)^(-0.4)·cos(5x) on [-1,1] | Algebraic blowup at x=±1 + oscillation | Slow: n=256→error ~1e-3 |
| Stress_NarrowPeak | exp(-1/(1-x²+1e-8)) on [-0.999,0.999] | Extremely steep gradients near ±1 | Already converged at n=64 (false alarm) |

### Convergence Data for Leg_Sing_Alg: (1-x²)^(-0.3)·exp(0.3x)
```
n=  32: 2.545898...   (error ~0.007 from n=512 value)
n=  64: 2.550292...   (error ~0.003)
n= 128: 2.551981...   (error ~0.001)
n= 256: 2.552625...   (error ~0.0003)
n= 512: 2.552870...   (best estimate, still not converged to tol=1e-12)
```

The error between n and 2n at n=256 is ~6×10⁻⁴ — far above tol=1e-12. Even at n=512 the gap is ~2×10⁻⁴. This is **O(n^(-p)) convergence** where p depends on singularity strength (α=0.3).

### Convergence Data for Cheb_Sing_Beta: (1-x²)^(-0.4)·cos(5x)
```
n=  32: -0.543664...   (error ~0.007 from n=512 value)
n=  64: -0.540532...   (error ~0.004)
n= 128: -0.539159...   (error ~0.001)
n= 256: -0.538559...   (error ~0.0003)
n= 512: -0.538297...   (best estimate, still not converged to tol=1e-12)
```

### Stress_NarrowPeak is NOT a Category C issue
The function exp(-1/(1-x²+1e-8)) converges perfectly at n=64:
```
n=  64: 0.443993...   (stable to 15 digits)
n= 128: 0.443993...   (identical)
```
The N/N flag here is a **false alarm** — the function has extremely steep gradients near ±1 but no singularity, and Gauss-Legendre resolves it fine at n=64. The convergence check fails because n2=128 gives essentially identical result, so err < tol should pass... unless the initial n used was too low (n=32→n2=64).

## Root Cause Analysis

### For Leg_Sing_Alg and Cheb_Sing_Beta:
The fundamental issue is that **Gauss-Legendre assumes smooth integrands**. When f(x) has an algebraic singularity at the endpoint, the derivatives blow up:
- f'(0.999) ≈ 2614 (vs f(0.999) ≈ 8.7)
- f''(0.999) ≈ 3.4×10⁶

This means the derivative growth probe classifies these as "polynomial" or worse, which correctly triggers higher n suggestions (lo=32, hi=128). However, even n=128 is insufficient for tol=1e-12 with algebraic convergence.

### Why These Currently Pass Fidelity Tests:
Both Direct and API return N/N with the same values — they agree on non-convergence. The fidelity test considers this a match (✅ YES). So Category C is **not about fixing fidelity mismatches** but about improving absolute accuracy for these pathological cases.

## Design Options

### Option 1: Weight Function Matching (Recommended)
For integrands of the form f(x) = g(x)·(1-x²)^(-α), extract the singular weight and use a quadrature rule designed for that weight. The remaining function g(x) is smooth, so convergence becomes exponential again.

**Implementation**: Detect algebraic endpoint singularities in `_find_singularities()`, then:
1. Extract singularity exponent α from expression structure (e.g., `(1-x^2)^(-0.3)` → α=0.3)
2. Use Gauss-Jacobi quadrature with parameters (α, α) instead of Gauss-Legendre
3. This requires adding a Jacobi quadrature module or using scipy.special.roots_jacobi

**Pros**: Mathematically elegant; exponential convergence restored
**Cons**: Requires new Jacobi quadrature implementation; singularity detection is heuristic

### Option 2: Variable Transformation (Simpler)
Apply a change of variable that removes the singularity. For f(x) = g(x)·(1-x²)^(-α):
- Use x = cos(t), dx = -sin(t)dt, which maps [-1,1] → [0,π]
- The (1-x²)^(-α) factor becomes sin^(-2α)(t), and the Jacobian contributes sin(t)
- Net weight: sin^(1-2α)(t) on [0,π], which is smooth for α < 0.5

**Pros**: No new quadrature module needed; uses existing Legendre on transformed interval
**Cons**: Requires symbolic manipulation to detect and apply transformation

### Option 3: Interval Splitting with Adaptive Refinement (Pragmatic)
Split the interval at midpoint, then recursively subdivide regions where convergence is slow. This concentrates nodes near singularities without changing the quadrature rule.

**Pros**: Works for any singularity type; no new modules needed
**Cons**: More complex control flow; may require many subdivisions

### Option 4: Honest Non-Convergence Reporting (Minimal Change)
Accept that algebraic singularities cannot converge to tol=1e-12 with standard Gauss-Legendre. Improve the `message` field to explain why, and optionally relax tol for these cases.

**Pros**: No code changes needed; current behavior is correct
**Cons**: Doesn't improve accuracy

## Recommended Approach: Option 1 (Weight Function Matching)

### Implementation Plan

#### Step 1: Add singularity exponent detection
Extend `_find_singularities()` to extract the algebraic exponent α for endpoint singularities. Look for patterns like `(x - a)^(-α)` or `(b - x)^(-α)` in the expression.

```python
def _extract_endpoint_singularity_exponents(self, expr, a, b):
    """Detect (x-a)^(-alpha) and (b-x)^(-alpha) factors.
    Returns dict: {'left': alpha_left, 'right': alpha_right}"""
```

#### Step 2: Add Gauss-Jacobi quadrature path
Use `scipy.special.roots_jacobi` to compute nodes/weights for Jacobi polynomials P_n^(α,β)(x), which are orthogonal with respect to (1-x)^α(1+x)^β on [-1,1].

```python
def _integrate_jacobi(self, func_stripped, alpha, beta, n):
    """Integrate g(x)*(1-x)^alpha*(1+x)^beta using Gauss-Jacobi quadrature.
    func_stripped = f(x) / [(1-x)^alpha * (1+x)^beta]"""
    from scipy.special import roots_jacobi
    nodes, weights = roots_jacobi(n, alpha, beta)
    return sum(w * func_stripped(xi) for xi, w in zip(nodes, weights))
```

#### Step 3: Modify execute_quadrature to use Jacobi when appropriate
When endpoint singularities with known exponents are detected, strip the weight and use Jacobi quadrature instead of Legendre.

### Files Modified
- `api_new/quadrature_analyzer_d_adapted.py` — add singularity exponent detection and Jacobi path

### Files NOT Modified (protected)
- `legendre/`, `chebyshev/`, `hermite/`, `laguerre/` — core quadrature modules untouched
- `api_new/main_mod4intv.py` — server middleware untouched

## Expected Impact

| Test | Current (n=128, N) | After Fix (Jacobi n=32) |
|------|-------------------|------------------------|
| Leg_Sing_Alg | 2.551981... (err ~0.001) | Exponential convergence expected |
| Cheb_Sing_Beta | -0.539159... (err ~0.001) | Exponential convergence expected |

## Risks & Limitations
- Singularity exponent detection is heuristic and may miss complex expressions
- scipy.special.roots_jacobi requires scipy dependency
- For α ≥ 1, the integral diverges — need to detect this case
- Stress_NarrowPeak will NOT be fixed by this approach (it's not a true singularity)