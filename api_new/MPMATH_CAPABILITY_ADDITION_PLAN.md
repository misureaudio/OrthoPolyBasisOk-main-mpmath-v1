# MPMATH CAPABILITY ADDITION PLAN — quadrature_analyzer_d_adapted_v2_G3.py

## Document Version: 1.0
## Date: 2026-05-15
## Target File: `api_new/quadrature_analyzer_d_adapted_v2_G3.py`
## Constraint: NO code modification — this is a PLAN only

---

## 1. CURRENT STATE ANALYSIS

### 1.1 What Already Exists (Partial mpmath Support)

The file already has **superficial** mpmath support that is incomplete and non-functional for its intended purpose:

| Location | Current State | Problem |
|----------|---------------|---------|
| Line 705 | `use_mpmath: bool = False` parameter in `execute_quadrature()` | Flag exists but only reaches Hermite family |
| Line 691 | `quad = GaussHermiteQuadrature(n={n_max}, use_mpmath=True)` in recommend_usage() | Documentation suggests mpmath for Hermite |
| Line 779-783 | Comment block "BEGIN GEMINI 3 MOD for mpmath" | Only limits n2 cap to 150, no actual mpmath logic added |
| Line 879 | `GaussHermiteQuadrature(n=n, use_mpmath=use_mpmath)` in `_integrate_hermite()` | Flag passed but module-level code casts mpf→float64 immediately (see hermite/integration.py line 58) |
| Line 827 | `LegendreQuadrature(n=n, use_mpmath=False)` in `_integrate_legendre()` | **Hardcoded False** — never uses mpmath |
| Line 1199 | `LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=False)` in `_integrate_laguerre()` | **Hardcoded False** — never uses mpmath |

### 1.2 The Three Layers of Precision Loss (Already Identified in MPMATH_IMPLEMENTATION_PLAN_v2.md)

| Layer | What Happens | Severity | Where It Occurs |
|-------|-------------|----------|-----------------|
| **L3: Integrand Evaluation** | `lambdify(..., modules="numpy")` evaluates in float64 | CRITICAL | All `_integrate_*` methods |
| **L2: Node/Weight Casting** | mpmath nodes computed at 80 dps, cast to `float64` | MEDIUM | Module-level integration files |
| **L1: Result Casting** | `float(value)` after quadrature | LOW | Analyzer + module return statements |

### 1.3 Complete Inventory of `use_mpmath` Propagation Gaps

| Method | Receives `use_mpmath`? | Passes to Sub-module? | Actual Effect |
|--------|----------------------|---------------------|---------------|
| `_integrate_legendre` (line 823) | **NO** — not in signature | Hardcoded `False` line 827 | **NEVER uses mpmath** |
| `_integrate_chebyshev` (line 832) | **NO** — not in signature | No parameter at all | **NEVER uses mpmath** |
| `_integrate_hermite` (line 860) | YES — line 861 | Passed to `GaussHermiteQuadrature` line 879 | Uses mpmath for nodes, but casts result to float at line 880 |
| `_integrate_laguerre` (line 1144) | **NO** — not in signature | Hardcoded `False` line 1199 | **NEVER uses mpmath** |
| `_integrate_jacobi` / `_integrate_with_jacobi_weight` | **NO** — not in signature | Uses `scipy.special.roots_jacobi` | **NO mpmath support exists** |

### 1.4 Complete Inventory of Premature `float()` Casts (that destroy mpmath precision)

| Line | Context | Code | Impact |
|------|---------|------|--------|
| 776 | Convergence check | `value_n = float(value)` | **DESTROYS mpmath precision** |
| 798 | Convergence check (2n) | `value_2n = float(value_2n)` | **DESTROYS mpmath precision** |
| 830 | `_integrate_legendre` return | `float(result) * scale` | Always casts, even if mpmath used internally |
| 841 | `_integrate_chebyshev` (Jacobi path) | `float(q.clenshaw_curtis_quadrature(...))` | Always casts |
| 843 | `_integrate_chebyshev` (standard path) | `float(clencurt_integrate_interval(...))` | Always casts |
| 880 | `_integrate_hermite` return | `float(quad.integrate(stripped))` | **Casts AFTER mpmath computation** |
| 1126 | `_integrate_jacobi` return | `float(total) * weight_scale` | Always casts |
| 1200 | `_integrate_laguerre` | `float(quad.integrate(stripped))` | Always casts |

---

## 2. REQUIRED MODIFICATIONS — COMPLETE PLAN

### PHASE 1: Module-Level Foundation (External Files)

**These changes are prerequisites but OUTSIDE the target file.** The analyzer cannot use mpmath properly until these module files provide full-precision methods.

#### 1A. File: `hermite/integration.py`
**Add method:** `integrate_mp(self, f_or_expr, dps=None)`
- Recomputes nodes/weights at target precision (bypasses float64 cast at line 58)
- Uses `lambdify(modules='mpmath')` for integrand evaluation
- Returns raw `mpmath.mpf` — NO float() cast

#### 1B. File: `laguerre/integration.py`
**Add method:** `integrate_mp(self, f_or_expr, dps=80)`
- Recomputes nodes/weights at target precision (bypasses float64 casts at lines 41-42)
- Uses `lambdify(modules='mpmath')` for integrand evaluation
- Returns raw `mpmath.mpf` — NO float() cast

#### 1C. File: `legendre/integration.py`
**Add method:** `integrate_mp(self, f_or_expr, a=-1.0, b=1.0)` to `LegendreQuadrature`
- Bridges to existing `HighPrecisionGaussLegendre` class (already provides full mpmath)
- Adds [a,b] interval transformation in mpmath arithmetic

#### 1D. File: `chebyshev/integration.py`
**Add method:** `integrate_on_interval_mp(self, f_or_expr, a, b, n=32, dps=80)`
- Bridges to existing `ClenshawCurtisMP` class in `integration_mp.py`

---

### PHASE 2: Analyzer File Modifications (`quadrature_analyzer_d_adapted_v2_G3.py`)

#### Change A1: Add mpmath import at module level (after line 37)
```python
import mpmath as mp
```

#### Change A2: Modify `_integrate_legendre` signature and implementation (line 823)

**Current:**
```python
def _integrate_legendre(self, func, a: float, b: float, n: int) -> float:
```

**New:**
```python
def _integrate_legendre(self, func_or_expr, a: float, b: float, n: int,
                        use_mpmath: bool = False):
    """Integrate on [a,b] using Gauss-Legendre.
    
    If use_mpmath=True, func_or_expr should be a sympy expression (not numpy callable).
    Returns float when use_mpmath=False, mp.mpf when True.
    """
    from legendre import LegendreQuadrature
    
    if use_mpmath:
        quad = LegendreQuadrature(n=n, use_mpmath=True, dps=80)
        return quad.integrate_mp(func_or_expr, a=a, b=b)  # returns mp.mpf
    else:
        scale = (b - a) / 2.0
        shift = (b + a) / 2.0
        quad = LegendreQuadrature(n=n, use_mpmath=False)
        transformed = lambda t: func_or_expr(scale * t + shift)
        return float(quad.integrate(transformed)) * scale
```

#### Change A3: Modify `_integrate_chebyshev` signature and implementation (line 832)

**Current:**
```python
def _integrate_chebyshev(self, func, expr, variable, a, b, n,
                         has_endpoint_singularity=False):
```

**New:**
```python
def _integrate_chebyshev(self, func_or_expr, expr, variable, a, b, n,
                         has_endpoint_singularity=False, use_mpmath=False):
    if use_mpmath:
        from chebyshev.integration_mp import ClenshawCurtisMP
        cc = ClenshawCurtisMP(n, dps=80)
        from sympy import Symbol, lambdify
        v = Symbol(variable)
        f_mp = lambdify(v, expr, modules='mpmath')
        return cc.integrate_on_interval(f_mp, a, b)  # returns mp.mpf
    else:
        from chebyshev import clencurt_integrate_interval, ChebyshevQuadrature
        if has_endpoint_singularity:
            from sympy import sqrt, Symbol, lambdify
            v = Symbol(variable)
            weight = 1 / sqrt(1 - v**2)
            stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")
            q = ChebyshevQuadrature()
            return float(q.clenshaw_curtis_quadrature(stripped, n=n))
        else:
            return float(clencurt_integrate_interval(func_or_expr, a, b, n))
```

#### Change A4: Modify `_integrate_hermite` for full mpmath pipeline (line 860)

**Current:** Uses `lambdify(modules="numpy")` even when `use_mpmath=True`, then casts result to float.

**New:**
```python
def _integrate_hermite(self, expr, variable: str, n: int,
                       use_mpmath: bool = False):
    from sympy import exp, Symbol, lambdify
    from hermite import GaussHermiteQuadrature
    
    v = Symbol(variable)
    
    if use_mpmath:
        quad = GaussHermiteQuadrature(n=n, use_mpmath=True, dps=80)
        weight = exp(-v**2)
        stripped_expr = (expr / weight).simplify()
        
        result_mp = quad.integrate_mp(stripped_expr, dps=80)  # returns mp.mpf
        
        if abs(result_mp) > mp.mpf('1e30'):
            print(f"DEBUG _integrate_hermite: mpmath result too large — falling back to finite-window Legendre")
            L = self._compute_effective_support(expr, variable)
            osc_safe_n = self._compute_oscillation_safe_n(
                expr, variable, -L, L, max(n * 2, 64)
            )
            return self._integrate_legendre(expr, -L, L, osc_safe_n, use_mpmath=True)
        
        return result_mp  # Return mp.mpf, NOT float(result_mp)
    else:
        stripped = lambdify(v, (expr / exp(-v**2)).simplify(), modules="numpy")
        test_vals = stripped(np.array([3.0, 5.0, 7.0]))
        if np.any(np.abs(test_vals) > 1e6):
            func_original = lambdify(v, expr, modules="numpy")
            L = self._compute_effective_support(expr, variable)
            osc_safe_n = self._compute_oscillation_safe_n(
                expr, variable, -L, L, max(n * 2, 64)
            )
            return self._integrate_legendre(func_original, -L, L, osc_safe_n)
        
        quad = GaussHermiteQuadrature(n=n, use_mpmath=False)
        return float(quad.integrate(stripped))
```

#### Change A5: Modify `_integrate_laguerre` signature and implementation (line 1144)

**Current:** No `use_mpmath` parameter. Hardcoded `False` at line 1199.

**New:** Add `use_mpmath: bool = False` parameter with full mpmath pipeline branch that uses `LaguerreQuadrature.integrate_mp()`.

#### Change A6: Modify `_integrate_jacobi` and `_integrate_with_jacobi_weight` for mpmath support (lines 1107, 1128)

**For `_integrate_jacobi`:** Add `use_mpmath: bool = False` parameter. When True, use `mpmath.quad()` instead of scipy's `roots_jacobi`.

**For `_integrate_with_jacobi_weight`:** Add `use_mpmath: bool = False` parameter. When True, pass sympy expression to mpmath quad path.

#### Change A7: Update `execute_quadrature` dispatch (lines 759-771)

**Current:** Only Hermite receives the flag; all others use hardcoded paths.

**New:** Propagate `use_mpmath` to ALL integration methods with conditional dispatch based on family type and whether sympy expression or numpy callable should be passed.

#### Change A8: Fix convergence checking for mixed types (lines 776-812)

**Current:** Always casts results to float before comparison, destroying mpmath precision.

**New:** When `use_mpmath=True`, keep values as mp.mpf throughout convergence check. Only cast to float at the final QuadratureResult construction.

---

### PHASE 3: Optional — Adaptive Fallback with mpmath.quad

Add `_integrate_mpmath_adaptive` method for extreme cases where Gaussian quadrature fails even with mpmath precision. Uses `mpmath.quad()` directly on the full integrand with adaptive subdivision.

---

## 3. IMPLEMENTATION ORDER (DEPENDENCY GRAPH)

```
PHASE 1: Module-Level Changes (prerequisites, outside target file)
    |-- hermite/integration.py      [H1] Add integrate_mp method
    |-- laguerre/integration.py     [G1] Add integrate_mp method  
    |-- legendre/integration.py     [L1] Add bridge to HighPrecisionGaussLegendre
    |-- chebyshev/integration.py    [C1] Add bridge to ClenshawCurtisMP

PHASE 2: Analyzer Refactoring (depends on Phase 1)
    |-- A1: mpmath import
    |-- A2: _integrate_legendre with use_mpmath parameter
    |-- A3: _integrate_chebyshev with use_mpmath parameter
    |-- A4: _integrate_hermite full mpmath pipeline
    |-- A5: _integrate_laguerre with use_mpmath parameter
    |-- A6: _integrate_jacobi with mpmath.quad fallback
    |-- A7: execute_quadrature dispatch propagation
    |-- A8: Convergence checking for mixed types (mpf vs float)

PHASE 3: Optional Adaptive Fallback
    |-- Add _integrate_mpmath_adaptive method
```

---

## 4. FILES TO MODIFY — COMPLETE INVENTORY

| # | File | Changes | Lines Affected |
|---|------|---------|---------------|
| 1 | `hermite/integration.py` | Add `integrate_mp` method | ~30 new lines |
| 2 | `laguerre/integration.py` | Add `integrate_mp` method | ~30 new lines |
| 3 | `legendre/integration.py` | Add `integrate_mp` bridge to `LegendreQuadrature` | ~15 new lines |
| 4 | `chebyshev/integration.py` | Add `integrate_on_interval_mp` bridge | ~15 new lines |
| 5 | `quadrature_analyzer_d_adapted_v2_G3.py` | A1-A8: Full analyzer refactoring | ~60 modified, ~40 new lines |

**Total estimated effort:** ~190 lines of code changes across 5 files.

---

## 5. EXPECTED BEHAVIOR AFTER FIX

### Before (broken):
```python
result = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=True)
# use_mpmath=True is accepted but:
#   - Legendre path: hardcoded False, uses numpy throughout
#   - Laguerre path: hardcoded False, uses numpy throughout  
#   - Jacobi path: no mpmath support at all
#   - Hermite path: nodes computed in mpf then cast to float64 (line 58)
#   - ALL paths: integrand evaluated via lambdify(modules="numpy") = float64 arithmetic
#   - ALL paths: result cast to float() at the end
# NET EFFECT: use_mpmath=True is essentially a no-op for precision.
```

### After (fixed):
```python
result = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=True)
# Full mpmath pipeline for ALL polynomial families:
#   - Nodes/weights computed and kept as mp.mpf at 80 dps (no float64 cast)
#   - Integrand evaluated via lambdify(modules="mpmath") = arbitrary precision
#     → exp(-100) returns a precise mpf number, NOT 0.0
#   - Summation in mpmath arithmetic
#   - Convergence check uses mpf comparison (no premature float cast)
#   - Final result cast to float() ONLY for QuadratureResult.value field
# NET EFFECT: Full 80-digit intermediate precision preserved throughout pipeline.
```

---

## 6. RISK ASSESSMENT AND MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Performance regression (mpmath is slow) | HIGH | MEDIUM | `use_mpmath` defaults to False; only activated explicitly by user |
| Breaking existing tests expecting float return types | MEDIUM | LOW | `QuadratureResult.value` remains float; internal mpf preserved until final cast in Change A8 |
| mpmath.quad for Jacobi may not converge for strong singularities | LOW | LOW | Alpha/beta < 1 ensures integrable singularities; fallback to numpy path available |
| Sympy `lambdify(modules='mpmath')` fails for some expressions | MEDIUM | MEDIUM | Add try/except with fallback: `lambda x: mp.mpf(f(float(x)))` |
| Fallback paths (Legendre on finite window) not mpmath-aware | LOW | MEDIUM | Change A4/A5 ensure recursive calls also use `use_mpmath=True` |

---

## 7. VERIFICATION STRATEGY

After implementation, verify with these test cases:

```python
# Test 1: Underflow case — exp(-100) should NOT be zero in mpmath path
analyzer.execute_quadrature("exp(-(x-5)**2)", interval=(0, 10), use_mpmath=True)

# Test 2: Compare precision between use_mpmath=True and False  
r_float = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=False)
r_mp    = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=True)

# Test 3: Infinite interval with algebraic decay (no Gaussian weight match)
analyzer.execute_quadrature("1 / (1 + x**4)", use_mpmath=True)