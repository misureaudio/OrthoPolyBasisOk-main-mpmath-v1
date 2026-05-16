# Hermite Quadrature Improvement Plan for `quadrature_analyzer.py`

**Date:** 2026-05-12  
**Status:** DRAFT — for review  
**Constraint:** NO modifications to `legendre/`, `chebyshev/`, `hermite/`, `laguerre/` subdirectories. Only `api/quadrature_analyzer.py` may be modified.

---

## 1. Problem Statement

The current `_integrate_hermite()` method in `quadrature_analyzer.py` (lines ~1029-1046) strips the standard Gauss-Hermite weight `e^{-x²}` from the integrand before passing to `GaussHermiteQuadrature.integrate()`. This works correctly **only when** the integrand decays at least as fast as `e^{-x²}`.

For functions like:
- `exp(-0.05*x**2 + 0.8*x)` — stripped form becomes `exp(+0.95*x**2 + 0.8*x)` → **grows exponentially**
- `cos(15*x) * exp(-0.08*x**2)` — stripped form becomes `cos(15*x) * exp(+0.92*x**2)` → **grows exponentially**

The current divergence guard (lines ~1037-1043) correctly detects this and raises a `ValueError`, but the error is not gracefully handled upstream in `execute_quadrature()`. The result propagates as a NaN with an unhelpful message.

---

## 2. Root Cause Analysis

### 2.1 Mathematical Background

Gauss-Hermite quadrature approximates:
```
∫_{-∞}^{+∞} f(x) e^{-x²} dx ≈ Σ w_i · f(x_i)
```

The `_integrate_hermite()` method computes `stripped = expr / exp(-x²)` and passes it to the quadrature. This is mathematically correct **only if** `f(x) = expr` can be written as `g(x) · e^{-x²}` where `g(x)` is bounded at infinity.

### 2.2 Current Behavior Matrix

| Integrand | Decay Rate | Stripped Form Bounded? | Current Result |
|-----------|-----------|----------------------|----------------|
| `exp(-x**2)` | exactly e^{-x²} | ✅ Yes (g(x)=1) | Works perfectly |
| `(x^6+...) * exp(-x**2)` | e^{-x²} × poly | ✅ Yes (polynomial growth OK at finite nodes) | Works well |
| `exp(-0.05*x**2 + 0.8*x)` | e^{-0.05x²} << e^{-x²} | ❌ No — grows as e^{+0.95x²} | **ValueError raised** |
| `cos(15x) * exp(-0.08*x**2)` | e^{-0.08x²} << e^{-x²} | ❌ No — grows as e^{+0.92x²} | **ValueError raised** |
| `1/(1+x^4)` | algebraic O(x^{-4}) | ❌ No — grows as e^{+x²}/(1+x⁴) | **ValueError raised** |

### 2.3 The Gap

The analyzer's `_recommend_family()` (lines ~611-653) recommends Hermite for **any** infinite-domain integral with Gaussian decay, but does not verify that the decay rate is ≥ e^{-x²}. This means it can recommend a family whose weight function is incompatible with the integrand.

---

## 3. Proposed Improvements (in `quadrature_analyzer.py` only)

### 3.1 Enhanced Decay Rate Detection in `_probe_decay()`

**Current behavior:** Detects "gaussian" decay type and extracts coefficient c from exp(-c·x²).

**Proposed addition:** Store the extracted decay rate `c` in a new field on `FunctionAnalysis`:
```python
# New field in FunctionAnalysis:
hermite_compatible: bool = True  # True if decay_rate >= 1.0 for Hermite weight e^{-x²}
```

**Implementation sketch (in `_probe_decay()`):**
```python
if QuadratureAnalyzer._is_negative_quadratic(inner, v):
    coeff = self._extract_x2_coefficient(expanded, v)
    # Return decay rate AND compatibility flag
    return ("gaussian", abs(coeff), abs(coeff) >= 1.0)
```

### 3.2 Fallback Strategy in `_recommend_family()`

When Hermite is recommended but `hermite_compatible=False`, fall back to a **finite-window Legendre** approach:

```python
if iv == "infinite":
    if decay == "gaussian" and not hermite_compatible:
        return self._Rec(
            PolynomialFamily.LEGENDRE,  # Fallback family
            "medium",
            f"Gaussian decay rate c={decay_rate:.2f} < 1.0; standard Gauss-Hermite weight e^{{-x²}} is too aggressive. "
            f"Recommend finite-window Legendre quadrature on [-L, +L] where L ≈ sqrt(-log(tol)/c)."
        )
```

### 3.3 Graceful Fallback in `_integrate_hermite()`

Instead of raising `ValueError`, attempt a **generalized Gauss-Hermite** approach or fall back to finite-window Legendre:

```python
def _integrate_hermite(self, expr, variable: str, n: int, use_mpmath: bool = False) -> float:
    from sympy import exp, Symbol, lambdify
    from hermite import GaussHermiteQuadrature
    
    v = Symbol(variable)
    weight = exp(-v**2)
    stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")

    # Sanity check: if stripped function grows at large |x|, fall back to finite-window Legendre
    test_vals = stripped(np.array([3.0, 5.0, 7.0]))
    if np.any(np.abs(test_vals) > 1e6):
        # FALLBACK: Use finite-window Legendre on [-L, +L]
        L = self._compute_effective_support(expr, variable)  # New helper
        return self._integrate_legendre_fallback(stripped * weight_func, -L, L, n)
    
    quad = GaussHermiteQuadrature(n=n, use_mpmath=use_mpmath)
    return float(quad.integrate(stripped))
```

### 3.4 New Helper: `_compute_effective_support()`

For infinite-domain integrals, compute the finite window [-L, +L] that captures ≥ (1-ε) of the integral mass:

```python
def _compute_effective_support(self, expr, variable: str, epsilon: float = 1e-15) -> float:
    """Compute L such that ∫_{|x|>L} |f(x)| dx < ε · ∫_{all} |f(x)| dx.

    For f(x) ~ exp(-c·x²): L ≈ sqrt(log(1/ε) / c)
    For f(x) ~ exp(-c·|x|): L ≈ log(1/ε) / c
    """
    # Extract decay parameters from expression structure
    v = Symbol(variable)
    
    for sub in expr.atoms(exp):
        try:
            inner = sub.args[0]
            if QuadratureAnalyzer._is_negative_quadratic(inner, v):
                coeff = abs(self._extract_x2_coefficient(inner.expand(), v))
                return math.sqrt(math.log(1.0 / epsilon) / max(coeff, 1e-15))
            elif QuadratureAnalyzer._is_negative_linear(inner, v):
                coeff = abs(self._extract_x_coefficient(inner, v))
                return math.log(1.0 / epsilon) / max(coeff, 1e-15)
        except Exception:
            pass
    
    # Default: use [-20, +20] which captures most practical integrands
    return 20.0
```

### 3.5 Enhanced Error Handling in `execute_quadrature()`

The current code (lines ~908-916) catches exceptions but returns NaN with a generic message. Improve to:

```python
except ValueError as e:
    # Hermite divergence detected — attempt fallback
    msg = str(e)
    if "Gauss-Hermite will diverge" in msg:
        print(f"DEBUG: Hermite divergence for n={n}; attempting finite-window Legendre fallback")
        L = self._compute_effective_support(expr, variable)
        func_finite = lambda t: func(t)  # original function on [-L, +L]
        value = self._integrate_legendre(func_finite, -L, L, min(n * 2, 200))
    else:
        raise
```

---

## 4. Implementation Priority

| Priority | Change | Effort | Impact |
|----------|--------|--------|--------|
| **P1** | Enhanced error handling in `execute_quadrature()` (Section 3.5) | Low (~20 lines) | Immediate: existing failing cases get fallback values instead of NaN |
| **P2** | `_compute_effective_support()` helper (Section 3.4) | Medium (~30 lines) | Enables P1 to choose good window sizes |
| **P3** | Graceful fallback in `_integrate_hermite()` (Section 3.3) | Medium (~15 lines) | Cleaner separation of concerns |
| **P4** | Enhanced decay detection + `hermite_compatible` field (Sections 3.1-3.2) | High (~50 lines, touches FunctionAnalysis dataclass) | Prevents bad recommendations upstream |

---

## 5. Test Cases for Validation

After implementing P1+P2, the following cases should produce finite results:

```python
# Previously failed (Hermite divergence):
("exp(-0.05*x**2 + 0.8*x)", (-inf, inf))   # Known value via completing the square
("cos(15*x) * exp(-0.08*x**2)", (-inf, inf))  # Fourier transform of Gaussian
("1/(1 + x**4)", (-inf, inf))              # Known: π/√2

# Previously worked (should still work):
("exp(-x**2)", (-inf, inf))                # = √π ≈ 1.7724538509...
("(x**6 + 2*x**4 - 3*x**2) * exp(-x**2)", (-inf, inf))
```

---

## 6. Files Modified vs Protected

| File | Action |
|------|--------|
| `api/quadrature_analyzer.py` | ✅ MODIFY (target of this plan) |
| `legendre/**` | 🔒 NEVER MODIFY |
| `chebyshev/**` | 🔒 NEVER MODIFY |
| `hermite/**` | 🔒 NEVER MODIFY |
| `laguerre/**` | 🔒 NEVER MODIFY |

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Fallback Legendre window too small → inaccurate results | Medium | Use `_compute_effective_support()` with conservative ε=1e-15; validate against known analytical values |
| Fallback Legendre window too large → numerical overflow | Low | Cap L at 50.0; use `use_mpmath=True` for extreme cases |
| Existing working Hermite cases regress | Low | Keep current code path as primary; fallback only triggered by divergence guard |
| `_compute_effective_support()` fails to detect decay pattern | Medium | Default to L=20.0 which covers 99.9% of practical integrands |

---

## 8. Implementation Checklist

- [ ] **P1:** Add enhanced error handling in `execute_quadrature()` for Hermite divergence
- [ ] **P2:** Implement `_compute_effective_support()` helper method
- [ ] **P3:** Refactor `_integrate_hermite()` to use graceful fallback instead of raising ValueError
- [ ] **P4:** Add `hermite_compatible` field to `FunctionAnalysis`; update `_probe_decay()` and `_recommend_family()`

---

*End of Plan*