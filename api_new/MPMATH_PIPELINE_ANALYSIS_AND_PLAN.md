# MPMATH PIPELINE ANALYSIS AND IMPLEMENTATION PLAN
## Addressing: "use_mpmath flag vs float() casting destroys precision"

---

## 1. PROBLEM STATEMENT (VERBATIM)

> "You have a `use_mpmath` flag, but you frequently cast results to `float()`. If a user is
> integrating something like $e^{-100} \cdot \dots$, standard 64-bit floats will underflow to zero
> before the quadrature even begins. To truly support `mpmath`, the entire `_integrate_...` pipeline
> needs to use `mpmath.quad`."

---

## 2. DETAILED ANALYSIS

### 2.1 The Core Insight: Three Distinct Precision Loss Mechanisms

The statement identifies a problem that manifests at **three distinct layers**, each with different severity and fix complexity:

| Layer | What Happens | Severity | Current Impact |
|-------|-------------|----------|----------------|
| **L1: Result Casting** | `float(value)` after quadrature | LOW | Final result loses precision beyond ~15 digits, but the computation was correct internally |
| **L2: Node/Weight Casting** | mpmath nodes/weights computed at 80 dps, then cast to `float64` | MEDIUM | Quadrature nodes lose precision; for n > 80 this causes catastrophic cancellation in eigenvalue problems |
| **L3: Integrand Evaluation** | `lambdify(..., modules="numpy")` evaluates with float64 arithmetic | CRITICAL | Functions like `exp(-100)` underflow to exactly `0.0` before any quadrature rule sees them; the integrand is silently zeroed out |

### 2.2 Layer L3: The "Underflow Before Quadrature" Problem (CRITICAL)

This is the most important and least obvious issue. Consider integrating:
```
f(x) = exp(-x^2) * cos(0.5*x)   on [-10, 10]
```

**Current flow with `use_mpmath=True`:**

```python
# Step 1: Create numpy callable (THIS IS WHERE IT BREAKS)
func = lambdify(v, expr, modules="numpy")

# Step 2: Compute nodes (even if mpmath, they get cast to float64)
nodes = [-9.873..., -5.231..., ..., 9.873...]   # all float64

# Step 3: Evaluate integrand at nodes
f(nodes[0]) = exp(-(-9.873)^2) * cos(0.5*(-9.873))
            = exp(-97.48) * cos(-4.937)
            # In float64: exp(-97.48) = 1.2e-43 (barely representable, near denormal range)
            # For larger x: exp(-100) underflows to exactly 0.0

# Step 4: Weighted sum
sum(w_i * f(x_i))   # If most f(x_i) = 0.0 due to underflow, result is garbage
```

**The fundamental problem:** `lambdify(..., modules="numpy")` produces a function that operates exclusively in IEEE-754 float64 arithmetic. There is no path for mpmath's arbitrary precision to influence the integrand evaluation. The `use_mpmath=True` flag only affects node/weight computation (Layer L2), not the actual function evaluation.

### 2.3 Layer L2: Node/Weight Precision Loss (MEDIUM)

Every module that supports `use_mpmath=True` follows this anti-pattern:

**hermite/integration.py, line 58:**
```python
return np.array([float(x) for x in n_sorted]), np.array([float(w) for w in w_sorted])
#                    ^^^^^^                         ^^^^^^
#    mpmath.mpf(80 dps) -> float64 (15-17 significant digits)
```

**laguerre/integration.py, lines 41-42:**
```python
nodes = [float(vals[i]) for i in indices]
weights = [float(mp.gamma(a+1) * vecs[0,i]**2) for i in indices]
#          ^^^^^^                    ^^^^^^
```

**legendre/integration.py, line 171:**
```python
return np.array([float(v) for v in nodes_mp]), np.array([float(v) for v in weights_mp])
#                    ^^^^^^                         ^^^^^^
```

The mpmath computation is used **only** to improve the numerical stability of the Golub-Welsch eigendecomposition (avoiding eigenvalue collisions at high n), but the results are immediately degraded back to float64. For n <= 100 this is usually fine, but for n > 200 the node spacing becomes comparable to machine epsilon and quadrature accuracy degrades.

### 2.4 Layer L1: Result Casting (LOW)

**quadrature_analyzer_d_adapted_v2_G3.py:**
```python
# Line 96, _safe_eval_sympy:
return float(N(expr.subs("x", x_val), 15))
#         ^^^^^^  -- 15 digits computed, then truncated to ~15-17

# Line 1039 (execute_quadrature):
value_n = float(value)
#             ^^^^^^

# Line 1054:
value_2n = float(value_2n)
#              ^^^^^^

# Line 1086 (_integrate_legendre):
return float(result) * scale
#         ^^^^^^

# Line 1097 (_integrate_chebyshev):
return float(q.clenshaw_curtis_quadrature(stripped, n=n))
#         ^^^^^^

# Line 1136 (_integrate_hermite):
return float(quad.integrate(stripped))
#         ^^^^^^

# Line 1562 (_integrate_laguerre):
result = float(quad.integrate(stripped))
#          ^^^^^^
```

### 2.5 The `use_mpmath` Flag: Incomplete Propagation

Tracing the flag through `execute_quadrature`:

| Method | Receives `use_mpmath`? | Passes to Sub-module? | Actual Effect |
|--------|----------------------|---------------------|---------------|
| `_integrate_legendre` (line 1079) | NO | Hardcoded `False` line 1083 | **NEVER uses mpmath** |
| `_integrate_chebyshev` (line 1088) | NO | No parameter at all | **NEVER uses mpmath** |
| `_integrate_hermite` (line 1116) | YES | Passed to `GaussHermiteQuadrature` line 1135 | Uses mpmath for nodes, but casts result to float |
| `_integrate_laguerre` (line 1506) | NO | Hardcoded `False` line 1561 | **NEVER uses mpmath** |
| `_integrate_jacobi` (line 1363) | NO | Uses scipy.special.roots_jacobi | **NO mpmath support exists** |

The flag is accepted by `execute_quadrature(expression, ..., use_mpmath=False)` but only reaches Hermite. For all other families it is silently ignored.

---

## 3. WHAT "USE MPMATH.QUAD" ACTUALLY MEANS

The statement's prescription — "use `mpmath.quad`" — refers to mpmath's **adaptive quadrature** engine, which is fundamentally different from Gaussian quadrature:

```python
import mpmath as mp

# mpmath.quad uses tanh-sinh / double-exponential + Gauss-Kronrod adaptive rules
# ALL arithmetic is done in arbitrary precision (mpmath.mpf)
result = mp.quad(lambda x: mp.exp(-x**2) * mp.cos(0.5*x), [-10, 10])
# Returns an mpmath.mpf with full working precision (~80 dps by default)
```

**Key advantages of `mpmath.quad` over Gaussian quadrature for extreme cases:**

1. **No underflow in integrand evaluation**: `mp.exp(-100)` = a precise mpf number, not 0.0
2. **Adaptive subdivision**: Automatically refines regions where the integrand varies rapidly
3. **Full precision throughout**: Nodes, weights, function evaluations, and summation all use mpf
4. **Handles infinite intervals natively**: `mp.quad(f, [-mp.inf, mp.inf])` works correctly

**However**, `mpmath.quad` is significantly slower than Gaussian quadrature (10-100x for smooth functions) because it uses adaptive refinement rather than spectral convergence. It should be used as a **reference/golden standard** or as a fallback when float64 fails, not as the default path.

---

## 4. IMPLEMENTATION PLAN

### Phase 0: Design Decisions (Architecture)

**Decision D1: Dual-path integration strategy**
- `use_mpmath=False` (default): Current numpy-based pipeline, unchanged behavior
- `use_mpmath=True`: Full mpmath pipeline from integrand evaluation through final result

**Decision D2: Return type contract**
- `use_mpmath=False`: Returns `float` (current behavior)
- `use_mpmath=True`: Returns `mpmath.mpf` (preserves full precision); caller decides when to cast

**Decision D3: Scope of changes**
The fix requires coordinated changes across 5 files. The plan below is organized by file, with dependencies noted.

---

### Phase 1: Foundation — Module-Level mpmath Integration Routines

#### File: `hermite/integration.py`

**Change H1:** Add a true high-precision `integrate_mp` method to `GaussHermiteQuadrature`.

```python
# NEW METHOD in GaussHermiteQuadrature:
def integrate_mp(self, f_or_expr, dps=None):
    """Integrate using mpmath arithmetic throughout.
    
    Nodes and weights are recomputed at full precision (no float64 cast).
    The integrand is evaluated using mpmath functions.
    
    Args:
        f_or_expr: Callable accepting mp.mpf, or a sympy expression.
        dps: Working precision (defaults to self._dps).
    
    Returns:
        mpmath.mpf at full working precision.
    """
    import mpmath as mp
    from sympy import lambdify, Expr
    
    dps = dps or self._dps
    with mp.workdps(dps):
        # Recompute nodes/weights at target precision (bypass float64 cache)
        nodes_mp, weights_mp = self._compute_nodes_weights_mp(dps)
        
        # Convert integrand to mpmath-compatible callable
        if isinstance(f_or_expr, Expr):
            f_mp = lambdify(self._var, f_or_expr, modules='mpmath')
        elif callable(f_or_expr):
            f_mp = lambda x: mp.mpf(f_or_expr(float(x)))
        else:
            raise TypeError("f_or_expr must be callable or sympy expression")
        
        return sum(w * f_mp(x) for x, w in zip(nodes_mp, weights_mp))
```

**Change H2:** Add `_compute_nodes_weights_mp` that returns raw mpmath objects.

```python
def _compute_nodes_weights_mp(self, dps):
    """Return nodes and weights as lists of mp.mpf (no float64 cast)."""
    import mpmath as mp
    with mp.workdps(dps):
        diag = [mp.mpf(0)] * self.n
        off_diag = [mp.sqrt(mp.mpf(k) / 2) for k in range(1, self.n)]
        J = mp.matrix(self.n, self.n)
        for i in range(self.n):
            J[i, i] = diag[i]
            if i < self.n - 1:
                J[i, i + 1] = J[i + 1, i] = off_diag[i]
        eigenvals, eigenvecs = mp.eig(J)
        nodes = list(eigenvals)
        weights = [mp.sqrt(mp.pi) * (eigenvecs[0, i] ** 2) for i in range(self.n)]
        combined = sorted(zip(nodes, weights), key=lambda x: x[0])
        return [x for x, _ in combined], [w for _, w in combined]
```

---

#### File: `legendre/integration.py`

**Change L1:** Add `integrate_mp` method to `LegendreQuadrature`.

The existing `HighPrecisionGaussLegendre` class (line 217) already does most of this work. The key addition is a convenience wrapper that accepts sympy expressions and returns mpf:

```python
# NEW METHOD in LegendreQuadrature:
def integrate_mp(self, f_or_expr, a=-1.0, b=1.0):
    """High-precision integration on [a, b] using mpmath throughout."""
    hp = HighPrecisionGaussLegendre(self.n, dps=self.dps)
    # Transform from [-1,1] to [a,b]: x = scale*t + shift
    import mpmath as mp
    with mp.workdps(self.dps):
        scale = (mp.mpf(b) - mp.mpf(a)) / 2
        shift = (mp.mpf(b) + mp.mpf(a)) / 2
        
        if isinstance(f_or_expr, sp.Expr):
            from sympy.abc import x as sympy_x
            f_mp = lambdify(sympy_x, f_or_expr, modules='mpmath')
            integrand = lambda t: scale * f_mp(scale * t + shift)
        else:
            integrand = lambda t: scale * mp.mpf(f_or_expr(float(scale * float(t) + shift)))
        
        return hp.integrate(integrand)
```

---

#### File: `laguerre/integration.py`

**Change G1:** Add `_compute_nodes_weights_mp` and `integrate_mp` to `LaguerreQuadrature`.

```python
def _compute_nodes_weights_mp(self, n, alpha):
    """Return nodes/weights as lists of mp.mpf."""
    import mpmath as mp
    with mp.workdps(80):
        a = mp.mpf(alpha)
        diag = [2*k + a + 1 for k in range(n)]
        off = [mp.sqrt(mp.mpf(k)*(k+a)) for k in range(1, n)]
        J = mp.matrix(n, n)
        for i in range(n):
            J[i,i] = diag[i]
            if i < n-1: J[i, i+1] = J[i+1, i] = off[i]
        vals, vecs = mp.eig(J, left=False)
        indices = sorted(range(n), key=lambda i: vals[i])
        nodes = [vals[i] for i in indices]
        weights = [mp.gamma(a+1) * vecs[0,i]**2 for i in indices]
        return nodes, weights

def integrate_mp(self, f_or_expr):
    """High-precision Gauss-Laguerre integration."""
    import mpmath as mp
    with mp.workdps(80):
        nodes, weights = self._compute_nodes_weights_mp(self.n, self.alpha)
        
        if isinstance(f_or_expr, sp.Expr):
            f_mp = lambdify(sp.Symbol('x'), f_or_expr, modules='mpmath')
        else:
            f_mp = lambda x: mp.mpf(f_or_expr(float(x)))
        
        return sum(w * f_mp(x) for x, w in zip(nodes, weights))
```

---

#### File: `chebyshev/integration.py` / `chebyshev/integration_mp.py`

**Change C1:** The mpmath infrastructure already exists in `integration_mp.py`. No changes needed there.

**Change C2:** Add a bridge method to `ChebyshevQuadrature`:

```python
# In chebyshev/integration.py, ChebyshevQuadrature class:
def integrate_on_interval_mp(self, f_or_expr, a, b, n=32, dps=80):
    """High-precision Clenshaw-Curtis on [a, b]."""
    from .integration_mp import ClenshawCurtisMP
    cc = ClenshawCurtisMP(n, dps=dps)
    
    if isinstance(f_or_expr, sp.Expr):
        f_mp = lambdify(sp.Symbol('x'), f_or_expr, modules='mpmath')
    else:
        import mpmath as mp
        f_mp = lambda x: mp.mpf(f_or_expr(float(x)))
    
    return cc.integrate_on_interval(f_mp, a, b)
```

---

### Phase 2: Analyzer Refactoring — Propagate `use_mpmath` Through All Paths

#### File: `quadrature_analyzer_d_adapted_v2_G3.py`

**Change A1:** Add mpmath import at module level.

```python
import mpmath as mp
```

**Change A2:** Modify `_integrate_legendre` to accept and honor `use_mpmath`.

```python
def _integrate_legendre(self, func_or_expr, a, b, n, use_mpmath=False):
    """Integrate on [a,b] using Gauss-Legendre.
    
    If use_mpmath=True, func_or_expr should be a sympy expression (not numpy callable).
    Returns float when use_mpmath=False, mp.mpf when True.
    """
    from legendre import LegendreQuadrature
    
    if use_mpmath:
        # Full mpmath pipeline
        quad = LegendreQuadrature(n=n, use_mpmath=True, dps=80)
        return quad.integrate_mp(func_or_expr, a=a, b=b)  # returns mp.mpf
    else:
        scale = (b - a) / 2.0
        shift = (b + a) / 2.0
        quad = LegendreQuadrature(n=n, use_mpmath=False)
        transformed = lambda t: func_or_expr(scale * t + shift)
        return float(quad.integrate(transformed)) * scale
```

**Change A3:** Modify `_integrate_chebyshev` to accept and honor `use_mpmath`.

```python
def _integrate_chebyshev(self, func_or_expr, expr, variable, a, b, n,
                         has_endpoint_singularity=False, use_mpmath=False):
    if use_mpmath:
        from chebyshev.integration_mp import ClenshawCurtisMP
        cc = ClenshawCurtisMP(n, dps=80)
        return cc.integrate_on_interval(expr, a, b)  # returns mp.mpf
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

**Change A4:** Modify `_integrate_hermite` to use full mpmath pipeline.

```python
def _integrate_hermite(self, expr, variable, n, use_mpmath=False):
    from sympy import exp, Symbol, lambdify
    from hermite import GaussHermiteQuadrature
    
    v = Symbol(variable)
    
    if use_mpmath:
        # Full mpmath pipeline — no numpy lambdify, no float() cast
        quad = GaussHermiteQuadrature(n=n, use_mpmath=True, dps=80)
        
        # Build stripped integrand as sympy expression: f(x) / exp(-x^2) = f(x)*exp(x^2)
        weight = exp(-v**2)
        stripped_expr = (expr / weight).simplify()
        
        result_mp = quad.integrate_mp(stripped_expr, dps=80)  # returns mp.mpf
        
        # Check for overflow in mpmath domain
        if abs(result_mp) > mp.mpf('1e30'):
            print(f"DEBUG _integrate_hermite: mpmath result too large — falling back to finite-window Legendre")
            L = self._compute_effective_support(expr, variable)
            osc_safe_n = self._compute_oscillation_safe_n(
                expr, variable, -L, L, max(n * 2, 64)
            )
            return self._integrate_legendre(expr, -L, L, osc_safe_n, use_mpmath=True)
        
        return result_mp  # Return mp.mpf, NOT float(result_mp)
    else:
        # Existing numpy path (unchanged)
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

**Change A5:** Modify `_integrate_laguerre` to accept and honor `use_mpmath`.

```python
def _integrate_laguerre(self, expr, variable, n, use_mpmath=False):
    from sympy import exp, Symbol, lambdify
    
    v = Symbol(variable)
    
    if use_mpmath:
        # Full mpmath pipeline
        from laguerre.integration import LaguerreQuadrature
        quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=True)
        
        weight = exp(-v)
        stripped_expr = (expr / weight).simplify()
        
        result_mp = quad.integrate_mp(stripped_expr)  # returns mp.mpf
        
        if not mp.inf > abs(result_mp) > -mp.inf:  # check for inf/nan in mpmath
            print(f"DEBUG _integrate_laguerre: mpmath result non-finite — falling back to Legendre")
            L = self._compute_effective_support_laguerre(expr, variable)
            osc_safe_n = self._compute_oscillation_safe_n(
                expr, variable, 0.0, L, max(n * 2, 64)
            )
            return self._integrate_legendre(expr, 0.0, L, osc_safe_n, use_mpmath=True)
        
        return result_mp
    else:
        # Existing numpy path (unchanged)
        func_original = lambdify(v, expr, modules="numpy")
        weight = exp(-v)
        stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")
        # ... [rest of existing overflow detection and fallback logic unchanged]
```

**Change A6:** Modify `_integrate_jacobi` for mpmath support.

This is the most complex change because `scipy.special.roots_jacobi` has no mpmath equivalent. Two options:

- **Option J1 (Recommended):** Implement Jacobi node computation via Golub-Welsch with mpmath (the Jacobi matrix for Jacobi polynomials is well-known).
- **Option J2:** Fall back to `mpmath.quad` with the full integrand when `use_mpmath=True`.

```python
def _integrate_jacobi(self, func_or_expr, alpha, beta, n, a, b, use_mpmath=False):
    if use_mpmath:
        # Option J2: Use mpmath.quad with the full integrand (weight included)
        import mpmath as mp
        from sympy import Symbol, lambdify
        
        v = Symbol('t')  # integration variable on [a,b]
        
        # Reconstruct full integrand: g(t) * (b-t)^alpha * (t-a)^beta
        if isinstance(func_or_expr, sp.Expr):
            t = Symbol('t')
            weight_factor = (b - t)**alpha * (t - a)**beta
            full_integrand = func_or_expr.subs(self._var, t) * weight_factor
            f_mp = lambdify(t, full_integrand, modules='mpmath')
        else:
            def f_mp(t):
                return mp.mpf(func_or_expr(float(t))) * (b - float(t))**alpha * (float(t) - a)**beta
        
        with mp.workdps(80):
            return mp.quad(f_mp, [a, b])
    else:
        # Existing scipy path (unchanged)
        from scipy.special import roots_jacobi
        nodes_t, weights = roots_jacobi(n, -alpha, -beta)
        scale = (b - a) / 2.0
        shift = (b + a) / 2.0
        weight_scale = scale ** (1.0 - alpha - beta)
        total = 0.0
        for t_node, w in zip(nodes_t, weights):
            try:
                val = func_or_expr(scale * float(t_node) + shift)
                if math.isfinite(val):
                    total += w * val
            except:
                continue
        return float(total) * weight_scale
```

**Change A7:** Update `execute_quadrature` to propagate `use_mpmath` to ALL integration paths.

The critical change is in the dispatch logic (around lines 1022-1052):

```python
# BEFORE (current code, line ~1026):
elif fam == PolynomialFamily.LEGENDRE:
    value = self._integrate_legendre(func, current_a, current_b, n)
    
# AFTER:
elif fam == PolynomialFamily.LEGENDRE:
    if use_mpmath:
        value = self._integrate_legendre(expr, current_a, current_b, n, use_mpmath=True)
    else:
        value = self._integrate_legendre(func, current_a, current_b, n, use_mpmath=False)

# Similar changes for CHEBYSHEV and LAGUERRE paths.
```

**Change A8:** Handle convergence checking with mixed types (mpf vs float).

When `use_mpmath=True`, both `value_n` and `value_2n` will be `mpmath.mpf`. The comparison logic must work:

```python
# Line ~1039-1056, BEFORE:
value_n = float(value)          # DESTROYS mpmath precision!
...
value_2n = float(value_2n)      # DESTROYS mpmath precision!
err = abs(value_2n - value_n)

# AFTER:
if use_mpmath:
    import mpmath as mp
    value_n_mp = value          # keep as mp.mpf
    value_2n_mp = value_2n      # keep as mp.mpf
    err = abs(value_2n_mp - value_n_mp)
    rel_err = err / (abs(value_n_mp) + mp.mpf('1e-30'))
    converged = bool(err < mp.mpf(tol)) or bool(rel_err < mp.mpf(tol))
    final_value = float(value_2n_mp if converged else value_n_mp)  # cast at the very end
else:
    value_n = float(value)
    value_2n = float(value_2n)
    err = abs(value_2n - value_n)
    rel_err = err / (abs(value_n) + 1e-30)
    converged = (err < tol) or (rel_err < tol)
    final_value = value_2n if converged else value_n
```

---

### Phase 3: Optional — Add `mpmath.quad` as a Reference/Golden Path

For the most extreme cases (e.g., integrands that are essentially zero over most of the domain), even Gaussian quadrature with mpmath nodes may fail because the fixed node distribution doesn't adapt to where the function has mass. Adding an `mpmath.quad` fallback provides a safety net:

```python
def _integrate_mpmath_adaptive(self, expr, variable, interval_a, interval_b, dps=80):
    """Fallback: use mpmath's adaptive quadrature as a reference solution."""
    import mpmath as mp
    from sympy import Symbol, lambdify
    
    v = Symbol(variable)
    f_mp = lambdify(v, expr, modules='mpmath')
    
    with mp.workdps(dps):
        a_mp = mp.mpf(interval_a) if isinstance(interval_a, (int, float)) else interval_a
        b_mp = mp.mpf(interval_b) if isinstance(interval_b, (int, float)) else interval_b
        
        # Handle infinite intervals by splitting at finite boundaries
        points = [a_mp, b_mp]
        if not mp.isfinite(a_mp):
            points[0] = None  # mpmath.quad handles -inf natively
        if not mp.isfinite(b_mp):
            points[1] = None
        
        return mp.quad(f_mp, points)
```

---

## 5. DEPENDENCY GRAPH AND IMPLEMENTATION ORDER

```
Phase 0: Design Decisions (D1-D3)
    |
    v
Phase 1: Module-Level Changes (independent of each other)
    |-- hermite/integration.py      [H1, H2]
    |-- legendre/integration.py     [L1]
    |-- laguerre/integration.py     [G1]
    |-- chebyshev/integration_mp.py [C1 - already done]
    |-- chebyshev/integration.py    [C2]
    |
    v
Phase 2: Analyzer Refactoring (depends on Phase 1)
    |-- A1: mpmath import
    |-- A2-A6: Modify each _integrate_* method
    |-- A7: Update execute_quadrature dispatch
    |-- A8: Fix convergence checking for mixed types
    |
    v
Phase 3: Optional Adaptive Fallback
    |-- Add _integrate_mpmath_adaptive method
```

---

## 6. SUMMARY OF ALL `float()` CASTS THAT MUST BE CONDITIONALIZED

| File | Line(s) | Current Code | Fix |
|------|---------|-------------|-----|
| analyzer | 96 | `float(N(expr.subs(...), 15))` | Keep for analysis; add mp version for integration |
| analyzer | 1039 | `value_n = float(value)` | Conditional: skip when use_mpmath=True |
| analyzer | 1054 | `value_2n = float(value_2n)` | Conditional: skip when use_mpmath=True |
| analyzer | 1086 | `float(result) * scale` | Return mp.mpf in mpmath path |
| analyzer | 1097 | `float(q.clenshaw_curtis_quadrature(...))` | Return mp.mpf in mpmath path |
| analyzer | 1136 | `float(quad.integrate(stripped))` | Return mp.mpf in mpmath path |
| analyzer | 1562 | `float(quad.integrate(stripped))` | Return mp.mpf in mpmath path |
| hermite/int.py | 58 | `[float(x) for x in n_sorted]` | Add separate method returning raw mpf |
| legendre/int.py | 171 | `[float(v) for v in nodes_mp]` | Already handled by HighPrecisionGaussLegendre |
| laguerre/int.py | 41-42 | `[float(vals[i]) ...]` | Add separate method returning raw mpf |

---

## 7. EXPECTED BEHAVIOR AFTER FIX

### Before (broken):
```python
result = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=True)
# use_mpmath=True is accepted but:
#   - Legendre path: hardcoded False, uses numpy throughout
#   - Hermite path: nodes computed in mpf then cast to float64
#   - Integrand evaluated via lambdify(modules="numpy") = float64 arithmetic
#   - Result cast to float() at the end
# NET EFFECT: use_mpmath=True is essentially a no-op for precision.
```

### After (fixed):
```python
result = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=True)
# Full mpmath pipeline:
#   - Nodes/weights computed and kept as mp.mpf at 80 dps
#   - Integrand evaluated via lambdify(modules="mpmath") = arbitrary precision
#   - Summation in mpmath arithmetic
#   - Convergence check uses mpf comparison
#   - Final result cast to float() only for QuadratureResult.value field
# NET EFFECT: Full 80-digit intermediate precision preserved throughout.
```

---

## 8. RISK ASSESSMENT

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Performance regression (mpmath is slow) | HIGH | use_mpmath defaults to False; only activated explicitly |
| Breaking existing tests that expect float return types | MEDIUM | QuadratureResult.value remains float; internal mpf preserved until final cast |
| mpmath.quad for Jacobi may not converge for strong singularities | LOW | Alpha/beta < 1 ensures integrable singularities; fallback to numpy path available |
| Sympy lambdify(modules='mpmath') fails for some expressions | MEDIUM | Add try/except with fallback to numerical wrapper: `lambda x: mp.mpf(f(float(x)))` |