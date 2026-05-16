# MPMATH PIPELINE IMPLEMENTATION PLAN — Detailed Analysis & Actionable Steps

## Document Version: 2.0
## Date: 2026-05-14
## Target File: `quadrature_analyzer_d_adapted_v2_G3.py`
## Related Modules: `hermite/`, `legendre/`, `laguerre/`, `chebyshev/`

---

## 1. PROBLEM STATEMENT

> "You have a `use_mpmath` flag, but you frequently cast results to `float()`. If a user is
> integrating something like $e^{-100} \cdot \dots$, standard 64-bit floats will underflow to zero
> before the quadrature even begins. To truly support `mpmath`, the entire `_integrate_...` pipeline
> needs to use `mpmath.quad`."

---

## 2. COMPREHENSIVE CODEBASE ANALYSIS

### 2.1 The Three Layers of Precision Loss

The statement identifies a problem that manifests at **three distinct layers**, each with different severity:

| Layer | What Happens | Severity | Where It Occurs |
|-------|-------------|----------|-----------------|
| **L3: Integrand Evaluation** | `lambdify(..., modules="numpy")` evaluates in float64 | CRITICAL | All `_integrate_*` methods |
| **L2: Node/Weight Casting** | mpmath nodes computed at 80 dps, cast to `float64` | MEDIUM | Module-level integration files |
| **L1: Result Casting** | `float(value)` after quadrature | LOW | Analyzer + module return statements |

### 2.2 Layer L3 — The "Underflow Before Quadrature" Problem (CRITICAL)

This is the most important and least obvious issue. Consider integrating:

```
f(x) = exp(-x^2) * cos(0.5*x)   on [-10, 10]
```

**Current flow even with `use_mpmath=True`:**

```python
# Step 1: Create numpy callable (THIS IS WHERE IT BREAKS)
func = lambdify(v, expr, modules="numpy")

# Step 2: Evaluate integrand at nodes using float64 arithmetic
f(nodes[0]) = exp(-(-9.873)^2) * cos(0.5*(-9.873))
             # In float64: exp(-97.48) ≈ 1.2e-43 (near denormal range)
             # For larger x: exp(-100) underflows to exactly 0.0

# Step 3: Weighted sum — if most f(x_i) = 0.0, result is garbage
sum(w_i * f(x_i))
```

**The fundamental problem:** `lambdify(..., modules="numpy")` produces a function that operates exclusively in IEEE-754 float64 arithmetic. There is no path for mpmath's arbitrary precision to influence the integrand evaluation.

### 2.3 Layer L2 — Node/Weight Precision Loss (MEDIUM)

Every module that supports `use_mpmath=True` follows this anti-pattern:

**`hermite/integration.py`, line 58:**
```python
return np.array([float(x) for x in n_sorted]), np.array([float(w) for w in w_sorted])
#                    ^^^^^^                         ^^^^^^
#    mpmath.mpf(80 dps) -> float64 (15-17 significant digits)
```

**`laguerre/integration.py`, lines 41-42:**
```python
nodes = [float(vals[i]) for i in indices]
weights = [float(mp.gamma(a+1) * vecs[0,i]**2) for i in indices]
#          ^^^^^^                    ^^^^^^
```

**`legendre/integration.py`, line 171:**
```python
return np.array([float(v) for v in nodes_mp]), np.array([float(v) for v in weights_mp])
#                    ^^^^^^                         ^^^^^^
```

### 2.4 Layer L1 — Result Casting (LOW but Pervasive)

Complete inventory of `float()` casts in the analyzer file:

| Line | Context | Code | Impact |
|------|---------|------|--------|
| 96 | `_safe_eval_sympy` | `float(N(expr.subs("x", x_val), 15))` | Analysis only, acceptable |
| 776 | Convergence check | `value_n = float(value)` | **DESTROYS mpmath precision** |
| 791 | Convergence check | `value_2n = float(value_2n)` | **DESTROYS mpmath precision** |
| 823 | `_integrate_legendre` return | `float(result) * scale` | Always casts, even if mpmath used internally |
| 834 | `_integrate_chebyshev` | `float(q.clenshaw_curtis_quadrature(...))` | Always casts |
| 836 | `_integrate_chebyshev` | `float(clencurt_integrate_interval(...))` | Always casts |
| 873 | `_integrate_hermite` return | `float(quad.integrate(stripped))` | **Casts AFTER mpmath computation** |
| 1119 | `_integrate_jacobi` return | `float(total) * weight_scale` | Always casts |
| 1193 | `_integrate_laguerre` | `float(quad.integrate(stripped))` | Always casts |

### 2.5 The `use_mpmath` Flag: Incomplete Propagation

Tracing the flag through `execute_quadrature`:

| Method | Receives `use_mpmath`? | Passes to Sub-module? | Actual Effect |
|--------|----------------------|---------------------|---------------|
| `_integrate_legendre` (line 816) | **NO** — not in signature | Hardcoded `False` line 820 | **NEVER uses mpmath** |
| `_integrate_chebyshev` (line 825) | **NO** — not in signature | No parameter at all | **NEVER uses mpmath** |
| `_integrate_hermite` (line 853) | YES — line 854 | Passed to `GaussHermiteQuadrature` line 872 | Uses mpmath for nodes, but casts result to float at line 873 |
| `_integrate_laguerre` (line 1137) | **NO** — not in signature | Hardcoded `False` line 1192 | **NEVER uses mpmath** |
| `_integrate_jacobi` (line 1100) | **NO** — not in signature | Uses `scipy.special.roots_jacobi` | **NO mpmath support exists** |

The flag is accepted by `execute_quadrature(expression, ..., use_mpmath=False)` at line 705 but only reaches Hermite. For all other families it is silently ignored.

### 2.6 Current Module-Level Capabilities Assessment

| Module | Has mpmath node/weight computation? | Returns raw mpf or casts to float? | Has `integrate_mp` method? |
|--------|-----------------------------------|----------------------------------|--------------------------|
| `hermite/integration.py` | YES (`_compute_golub_welsch_mp`) | **CASTS** at line 58 | NO — only `.integrate()` which uses float64 arrays |
| `legendre/integration.py` | YES (`_golub_welsch_mp`, `HighPrecisionGaussLegendre`) | `HighPrecisionGaussLegendre.integrate()` returns raw mpf at line 261 | **YES** — `HighPrecisionGaussLegendre` class exists but is NOT used by analyzer |
| `laguerre/integration.py` | YES (`_compute_mp`) | **CASTS** at lines 41-42 | NO — only `.integrate()` which uses float64 arrays |
| `chebyshev/integration_mp.py` | YES (full infrastructure) | Returns raw mpf throughout | **YES** — `ClenshawCurtisMP.integrate_on_interval()` exists but is NOT used by analyzer |

---

## 3. WHAT "USE MPMATH.QUAD" MEANS IN PRACTICE

The statement's prescription — "use `mpmath.quad`" — refers to mpmath's **adaptive quadrature** engine:

```python
import mpmath as mp

# ALL arithmetic is done in arbitrary precision (mpmath.mpf)
result = mp.quad(lambda x: mp.exp(-x**2) * mp.cos(0.5*x), [-10, 10])
# Returns an mpmath.mpf with full working precision (~80 dps by default)
```

**Key advantages of `mpmath.quad` over Gaussian quadrature for extreme cases:**

1. **No underflow in integrand evaluation**: `mp.exp(-100)` = a precise mpf number, not 0.0
2. **Adaptive subdivision**: Automatically refines regions where the integrand varies rapidly
3. **Full precision throughout**: Nodes, weights, function evaluations, and summation all use mpf
4. **Handles infinite intervals natively**: `mp.quad(f, [-mp.inf, mp.inf])` works correctly

**However**, `mpmath.quad` is significantly slower than Gaussian quadrature (10-100x for smooth functions). It should be used as a **fallback/reference path**, not replacing the Gaussian quadrature entirely. The proper approach is:

- **Primary fix**: Make the existing Gaussian quadrature pipeline fully mpmath-aware (no float casts)
- **Secondary fix**: Add `mpmath.quad` as an adaptive fallback for cases where fixed-node Gaussian quadrature fails even with mpmath precision

---

## 4. IMPLEMENTATION PLAN

### Phase 0: Design Decisions

**D1: Dual-path integration strategy**
- `use_mpmath=False` (default): Current numpy-based pipeline, **unchanged behavior**
- `use_mpmath=True`: Full mpmath pipeline from integrand evaluation through final result

**D2: Return type contract within `_integrate_*` methods**
- `use_mpmath=False`: Returns `float`
- `use_mpmath=True`: Returns `mpmath.mpf` (preserves full precision internally)
- Final cast to `float` happens only in `execute_quadrature` when constructing `QuadratureResult`

**D3: Scope of changes — 5 files, coordinated**
1. `hermite/integration.py` — Add `integrate_mp` method
2. `laguerre/integration.py` — Add `integrate_mp` method
3. `legendre/integration.py` — Bridge to existing `HighPrecisionGaussLegendre`
4. `chebyshev/integration.py` — Bridge to existing `integration_mp.py` (already complete)
5. `quadrature_analyzer_d_adapted_v2_G3.py` — Propagate flag, add mpmath paths

---

### Phase 1: Module-Level Changes

#### 1A. File: `hermite/integration.py`

**Problem:** `_compute_golub_welsch_mp` computes nodes/weights at 80 dps precision but immediately casts them to float64 (line 58). The `.integrate()` method operates on these float64 arrays.

**Change H1: Add `integrate_mp` method to `GaussHermiteQuadrature`:**

```python
def integrate_mp(self, f_or_expr, dps=None):
    """Integrate using mpmath arithmetic throughout the pipeline.

    Nodes and weights are recomputed at full precision (no float64 cast).
    The integrand is evaluated using mpmath functions via lambdify(modules='mpmath').

    Args:
        f_or_expr: Callable accepting mp.mpf, or a sympy expression.
        dps: Working precision (defaults to self._dps).

    Returns:
        mpmath.mpf at full working precision — NO float() cast.
    """
    import mpmath as mp
    from sympy import lambdify, Expr, Symbol

    dps = dps or self._dps
    with mp.workdps(dps):
        # Recompute nodes/weights at target precision (bypass float64 cache)
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
        n_sorted, w_sorted = zip(*combined)

        # Convert integrand to mpmath-compatible callable
        if isinstance(f_or_expr, Expr):
            v = Symbol('x')
            f_mp = lambdify(v, f_or_expr, modules='mpmath')
        elif callable(f_or_expr):
            f_mp = lambda x: mp.mpf(f_or_expr(float(x)))
        else:
            raise TypeError("f_or_expr must be callable or sympy expression")

        # Summation in mpmath arithmetic — returns mp.mpf
        return sum(w * f_mp(x) for x, w in zip(n_sorted, w_sorted))
```

**Impact:** This method bypasses the float64 cache entirely. Nodes/weights are computed fresh at target precision and never cast to float. The integrand is evaluated via `lambdify(modules='mpmath')` which produces mpmath-compatible calls like `mp.exp()`, `mp.sin()`, etc.

---

#### 1B. File: `laguerre/integration.py`

**Problem:** `_compute_mp` casts results to float64 at lines 41-42. No `integrate_mp` method exists.

**Change G1: Add `integrate_mp` method to `LaguerreQuadrature`:**

```python
def integrate_mp(self, f_or_expr, dps=80):
    """High-precision Gauss-Laguerre integration with full mpmath pipeline."""
    import mpmath as mp
    from sympy import lambdify, Expr, Symbol

    with mp.workdps(dps):
        a = mp.mpf(self.alpha)
        # Recompute nodes/weights at target precision (no float64 cast)
        diag = [2*k + a + 1 for k in range(self.n)]
        off = [mp.sqrt(mp.mpf(k)*(k+a)) for k in range(1, self.n)]
        J = mp.matrix(self.n, self.n)
        for i in range(self.n):
            J[i,i] = diag[i]
            if i < self.n-1:
                J[i, i+1] = J[i+1, i] = off[i]
        vals, vecs = mp.eig(J, left=False)
        indices = sorted(range(self.n), key=lambda i: vals[i])
        nodes = [vals[i] for i in indices]
        weights = [mp.gamma(a+1) * vecs[0,i]**2 for i in indices]

        # Convert integrand to mpmath-compatible callable
        if isinstance(f_or_expr, Expr):
            v = Symbol('x')
            f_mp = lambdify(v, f_or_expr, modules='mpmath')
        elif callable(f_or_expr):
            f_mp = lambda x: mp.mpf(f_or_expr(float(x)))
        else:
            raise TypeError("f_or_expr must be callable or sympy expression")

        # Summation in mpmath arithmetic — returns mp.mpf
        return sum(w * f_mp(x) for x, w in zip(nodes, weights))
```

---

#### 1C. File: `legendre/integration.py`

**Status:** The `HighPrecisionGaussLegendre` class (line 217) already provides full mpmath integration with sympy expression support via `lambdify(modules='mpmath')`. Its `.integrate()` method returns raw `mp.mpf` at line 261.

**Change L1: Add convenience bridge to `LegendreQuadrature`:**

```python
def integrate_mp(self, f_or_expr, a=-1.0, b=1.0):
    """High-precision integration on [a, b] using mpmath throughout."""
    from .integration import HighPrecisionGaussLegendre  # already exists
    import mpmath as mp

    hp = HighPrecisionGaussLegendre(self.n, dps=self.dps)

    with mp.workdps(self.dps):
        scale_mp = (mp.mpf(b) - mp.mpf(a)) / 2
        shift_mp = (mp.mpf(b) + mp.mpf(a)) / 2

        # Transform integrand: integral_a^b f(x)dx = scale * integral_{-1}^{1} f(scale*t+shift) dt
        def transformed(t):
            return scale_mp * hp._evaluate(f_or_expr, scale_mp * t + shift_mp)

        return sum(w * transformed(x) for x, w in zip(hp.nodes, hp.weights))
```

**Note:** `HighPrecisionGaussLegendre.integrate()` already handles sympy expressions via lambdify. The bridge just adds the [a,b] interval transformation in mpmath arithmetic.

---

#### 1D. File: `chebyshev/integration.py`

**Status:** `chebyshev/integration_mp.py` already provides full infrastructure with `ClenshawCurtisMP.integrate_on_interval()` returning raw `mp.mpf`. No changes needed to the MP module itself.

**Change C1: Add bridge method to `ChebyshevQuadrature`:**

```python
def integrate_on_interval_mp(self, f_or_expr, a, b, n=32, dps=80):
    """High-precision Clenshaw-Curtis on [a, b]."""
    from .integration_mp import ClenshawCurtisMP
    from sympy import lambdify, Expr, Symbol

    cc = ClenshawCurtisMP(n, dps=dps)

    if isinstance(f_or_expr, Expr):
        v = Symbol('x')
        f_mp = lambdify(v, f_or_expr, modules='mpmath')
    else:
        import mpmath as mp
        f_mp = lambda x: mp.mpf(f_or_expr(float(x)))

    return cc.integrate_on_interval(f_mp, a, b)
```

---

### Phase 2: Analyzer Refactoring

#### File: `quadrature_analyzer_d_adapted_v2_G3.py`

**Change A1: Add mpmath import at module level (after line 37):**

```python
import mpmath as mp
```

---

**Change A2: Modify `_integrate_legendre` to accept and honor `use_mpmath`:**

Current signature (line 816):
```python
def _integrate_legendre(self, func, a: float, b: float, n: int) -> float:
```

New implementation:
```python
def _integrate_legendre(self, func_or_expr, a: float, b: float, n: int,
                        use_mpmath: bool = False):
    """Integrate on [a,b] using Gauss-Legendre.

    If use_mpmath=True, func_or_expr should be a sympy expression (not numpy callable).
    Returns float when use_mpmath=False, mp.mpf when True.
    """
    from legendre import LegendreQuadrature

    if use_mpmath:
        # Full mpmath pipeline — returns mp.mpf
        quad = LegendreQuadrature(n=n, use_mpmath=True, dps=80)
        return quad.integrate_mp(func_or_expr, a=a, b=b)
    else:
        scale = (b - a) / 2.0
        shift = (b + a) / 2.0
        quad = LegendreQuadrature(n=n, use_mpmath=False)
        transformed = lambda t: func_or_expr(scale * t + shift)
        return float(quad.integrate(transformed)) * scale
```

---

**Change A3: Modify `_integrate_chebyshev` to accept and honor `use_mpmath`:**

Current signature (line 825):
```python
def _integrate_chebyshev(self, func, expr, variable, a, b, n,
                         has_endpoint_singularity=False):
```

New implementation:
```python
def _integrate_chebyshev(self, func_or_expr, expr, variable, a, b, n,
                         has_endpoint_singularity=False, use_mpmath=False):
    if use_mpmath:
        from chebyshev.integration_mp import ClenshawCurtisMP
        cc = ClenshawCurtisMP(n, dps=80)
        # expr is a sympy expression; lambdify with mpmath modules
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

---

**Change A4: Modify `_integrate_hermite` to use full mpmath pipeline:**

Current implementation (lines 853-873): The `use_mpmath` flag is passed to `GaussHermiteQuadrature`, but the integrand is still created via `lambdify(modules="numpy")` at line 859, and the result is cast to `float()` at line 873.

New implementation:
```python
def _integrate_hermite(self, expr, variable: str, n: int,
                       use_mpmath: bool = False):
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

---

**Change A5: Modify `_integrate_laguerre` to accept and honor `use_mpmath`:**

Current signature (line 1137): No `use_mpmath` parameter. Hardcoded `False` at line 1192.

New implementation:
```python
def _integrate_laguerre(self, expr, variable: str, n: int,
                        use_mpmath: bool = False) -> float:
    from sympy import exp, Symbol, lambdify

    v = Symbol(variable)

    if use_mpmath:
        # Full mpmath pipeline
        from laguerre.integration import LaguerreQuadrature
        quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=True)

        weight = exp(-v)
        stripped_expr = (expr / weight).simplify()

        result_mp = quad.integrate_mp(stripped_expr, dps=80)  # returns mp.mpf

        if not mp.isfinite(result_mp):
            print(f"DEBUG _integrate_laguerre: mpmath result non-finite — falling back to Legendre")
            L = self._compute_effective_support_laguerre(expr, variable)
            osc_safe_n = self._compute_oscillation_safe_n(
                expr, variable, 0.0, L, max(n * 2, 64)
            )
            return self._integrate_legendre(expr, 0.0, L, osc_safe_n, use_mpmath=True)

        return result_mp  # Return mp.mpf, NOT float(result_mp)
    else:
        # Existing numpy path (unchanged — lines 1141-1203)
        func_original = lambdify(v, expr, modules="numpy")
        weight = exp(-v)
        stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")

        test_x = np.array([10.0, 30.0, 60.0])
        raw_vals = stripped(test_x)
        test_vals = np.atleast_1d(np.asarray(raw_vals, dtype=np.float64))

        if np.any(~np.isfinite(test_vals)):
            print(f"DEBUG _integrate_laguerre: stripped integrand overflows at probe points — falling back to finite-window Legendre")
            L = self._compute_effective_support_laguerre(expr, variable)
            osc_safe_n = self._compute_oscillation_safe_n(
                expr, variable, 0.0, L, max(n * 2, 64)
            )
            return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

        if len(test_vals) >= 2 and np.all(np.abs(test_vals) > 0):
            growth_ratio = abs(test_vals[-1]) / abs(test_vals[0])
            if growth_ratio > 1e6:
                print(f"DEBUG _integrate_laguerre: stripped integrand grows (ratio={growth_ratio:.2e}) — falling back to finite-window Legendre")
                L = self._compute_effective_support_laguerre(expr, variable)
                osc_safe_n = self._compute_oscillation_safe_n(
                    expr, variable, 0.0, L, max(n * 2, 64)
                )
                return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

        # Oscillation check (existing logic unchanged)
        omega_max = self._estimate_max_oscillation_frequency(expr, variable, 0.0, float("inf"))
        if omega_max > 1e-6:
            has_growth = False
            if len(test_vals) >= 2 and np.all(np.abs(test_vals) > 0):
                growth_ratio = abs(test_vals[-1]) / abs(test_vals[0])
                has_growth = growth_ratio > 1.5

            if has_growth:
                print(f"DEBUG _integrate_laguerre: oscillations + stripped growth detected — falling back to finite-window Legendre")
                L = self._compute_effective_support_laguerre(expr, variable)
                osc_safe_n = self._compute_oscillation_safe_n(
                    expr, variable, 0.0, L, max(n * 2, 64)
                )
                return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

        from laguerre import LaguerreQuadrature
        quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=False)
        result = float(quad.integrate(stripped))

        if not math.isfinite(result):
            print(f"DEBUG _integrate_laguerre: result is non-finite — falling back to finite-window Legendre")
            L = self._compute_effective_support_laguerre(expr, variable)
            osc_safe_n = self._compute_oscillation_safe_n(
                expr, variable, 0.0, L, max(n * 2, 64)
            )
            return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

        return result
```

---

**Change A6: Modify `_integrate_jacobi` for mpmath support:**

This is the most complex change because `scipy.special.roots_jacobi` has no mpmath equivalent. The recommended approach is **Option J2**: fall back to `mpmath.quad` with the full integrand when `use_mpmath=True`.

```python
def _integrate_jacobi(self, func_or_expr, alpha: float, beta: float, n: int,
                      a: float, b: float, use_mpmath: bool = False) -> float:
    if use_mpmath:
        # Use mpmath.quad with the full integrand (weight included).
        # This avoids needing Jacobi nodes at arbitrary precision.
        from sympy import Symbol, lambdify

        t = Symbol('t')  # integration variable on [a,b]

        if hasattr(func_or_expr, 'atoms'):  # it's a sympy expression
            # Reconstruct full integrand: g(t) * (b-t)^alpha * (t-a)^beta
            weight_factor = (b - t)**alpha * (t - a)**beta
            full_integrand = func_or_expr * weight_factor
            f_mp = lambdify(t, full_integrand, modules='mpmath')
        else:
            def f_mp(t):
                return mp.mpf(func_or_expr(float(t))) * (b - float(t))**alpha * (float(t) - a)**beta

        with mp.workdps(80):
            return mp.quad(f_mp, [a, b])  # returns mp.mpf
    else:
        # Existing scipy path (unchanged — lines 1101-1119)
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

---

**Change A7: Update `execute_quadrature` dispatch to propagate `use_mpmath`:**

Current code (lines 759-771) — the flag is accepted but only passed to Hermite:

```python
# CURRENT CODE (PROBLEMATIC):
if use_jacobi:
    value = self._integrate_with_jacobi_weight(expr, variable, n, current_a, current_b, left_alpha, right_beta)
elif fam == PolynomialFamily.LEGENDRE:
    value = self._integrate_legendre(func, current_a, current_b, n)  # NO use_mpmath!
elif fam == PolynomialFamily.CHEBYSHEV:
    value = self._integrate_chebyshev(func, expr, variable, current_a, current_b, n, ...)  # NO use_mpmath!
elif fam == PolynomialFamily.HERMITE:
    value = self._integrate_hermite(expr, variable, n, use_mpmath)  # Only this one gets it
else:  # LAGUERRE
    value = self._integrate_laguerre(expr, variable, n)  # NO use_mpmath!
```

New dispatch logic:
```python
if use_jacobi:
    if use_mpmath:
        value = self._integrate_with_jacobi_weight(expr, variable, n, current_a, current_b, left_alpha, right_beta, use_mpmath=True)
    else:
        value = self._integrate_with_jacobi_weight(expr, variable, n, current_a, current_b, left_alpha, right_beta, use_mpmath=False)
elif fam == PolynomialFamily.LEGENDRE:
    if use_mpmath:
        value = self._integrate_legendre(expr, current_a, current_b, n, use_mpmath=True)  # pass sympy expr
    else:
        value = self._integrate_legendre(func, current_a, current_b, n, use_mpmath=False)  # pass numpy callable
elif fam == PolynomialFamily.CHEBYSHEV:
    if use_mpmath:
        value = self._integrate_chebyshev(expr, expr, variable, current_a, current_b, n, use_mpmath=True)
    else:
        value = self._integrate_chebyshev(func, expr, variable, current_a, current_b, n, use_mpmath=False)
elif fam == PolynomialFamily.HERMITE:
    value = self._integrate_hermite(expr, variable, n, use_mpmath)  # already correct
else:  # LAGUERRE
    if use_mpmath:
        value = self._integrate_laguerre(expr, variable, n, use_mpmath=True)
    else:
        value = self._integrate_laguerre(expr, variable, n, use_mpmath=False)
```

---

**Change A8: Fix convergence checking for mixed types (mpf vs float):**

Current code (lines 776-805):
```python
# CURRENT CODE — DESTROYS mpmath precision:
value_n = float(value)          # Line 776: casts mp.mpf to float!
n2 = min(n * 2, 1000)
try:
    # ... compute value_2n ...
    value_2n = float(value_2n)  # Line 791: casts mp.mpf to float!
    err = abs(value_2n - value_n)
```

New convergence check:
```python
if use_mpmath:
    # Keep values as mp.mpf for precision-aware comparison
    value_n_mp = value          # keep as mp.mpf (no cast!)
    n2 = min(n * 2, 1000)
    try:
        # ... compute value_2n using mpmath paths ...

        value_2n_mp = value_2n  # keep as mp.mpf (no cast!)
        err = abs(value_2n_mp - value_n_mp)
        rel_err = err / (abs(value_n_mp) + mp.mpf('1e-30'))
        converged = bool(err < mp.mpf(tol)) or bool(rel_err < mp.mpf(tol))

        # Final cast to float ONLY for QuadratureResult.value field
        final_value = float(value_2n_mp if converged else value_n_mp)

        return QuadratureResult(
            value=final_value,
            family_used=fam,
            n_nodes=n2 if converged else n,
            converged=converged,
            error_estimate=float(err),
            message="Converged" if converged else f"Did not meet tol (err={float(err):.2e})"
        )
    except Exception as e:
        return QuadratureResult(value=float(value_n_mp), family_used=fam, n_nodes=n, converged=False, message=f"Check failed: {e}")
else:
    # Existing float path (unchanged)
    value_n = float(value)
    n2 = min(n * 2, 1000)
    try:
        # ... compute value_2n using numpy paths ...
        value_2n = float(value_2n)
        err = abs(value_2n - value_n)
        rel_err = err / (abs(value_n) + 1e-30)
        converged = (err < tol) or (rel_err < tol)

        return QuadratureResult(
            value=value_2n if converged else value_n,
            family_used=fam,
            n_nodes=n2 if converged else n,
            converged=converged,
            error_estimate=err,
            message="Converged" if converged else f"Did not meet tol (err={err:.2e})"
        )
    except Exception as e:
        return QuadratureResult(value=value_n, family_used=fam, n_nodes=n, converged=False, message=f"Check failed: {e}")
```

---

### Phase 3: Optional — `mpmath.quad` Adaptive Fallback

For the most extreme cases (integrands essentially zero over most of the domain), even Gaussian quadrature with mpmath nodes may fail because fixed node distribution doesn't adapt to where the function has mass.

**Add `_integrate_mpmath_adaptive` method:**

```python
def _integrate_mpmath_adaptive(self, expr, variable: str,
                                interval_a: float, interval_b: float,
                                dps: int = 80):
    """Fallback: use mpmath's adaptive quadrature as a reference solution.

    Uses tanh-sinh / double-exponential + Gauss-Kronrod adaptive rules.
    ALL arithmetic is done in arbitrary precision (mpmath.mpf).

    Handles infinite intervals natively via mp.quad(f, [-inf, inf]).
    """
    from sympy import Symbol, lambdify

    v = Symbol(variable)
    f_mp = lambdify(v, expr, modules='mpmath')

    with mp.workdps(dps):
        # Convert bounds to mpmath (handles inf natively)
        if math.isfinite(interval_a):
            a_mp = mp.mpf(interval_a)
        else:
            a_mp = interval_a  # -inf stays as is; mp.quad handles it

        if math.isfinite(interval_b):
            b_mp = mp.mpf(interval_b)
        else:
            b_mp = interval_b  # +inf stays as is

        return mp.quad(f_mp, [a_mp, b_mp])  # returns mp.mpf
```

This method should be called when the primary mpmath Gaussian quadrature path produces a non-finite result or when convergence checking fails.

---

## 5. DEPENDENCY GRAPH AND IMPLEMENTATION ORDER

```
Phase 0: Design Decisions (D1-D3) — DOCUMENT ONLY, no code changes
    |
    v
Phase 1: Module-Level Changes (independent of each other, can be done in parallel)
    |-- hermite/integration.py      [H1] Add integrate_mp method
    |-- laguerre/integration.py     [G1] Add integrate_mp method
    |-- legendre/integration.py     [L1] Add bridge to HighPrecisionGaussLegendre
    |-- chebyshev/integration.py    [C1] Add bridge to integration_mp (already exists)
    |
    v  (ALL Phase 1 changes must be complete before proceeding)
Phase 2: Analyzer Refactoring (depends on Phase 1)
    |-- A1: mpmath import
    |-- A2: _integrate_legendre with use_mpmath parameter
    |-- A3: _integrate_chebyshev with use_mpmath parameter
    |-- A4: _integrate_hermite full mpmath pipeline (fix lambdify + float cast)
    |-- A5: _integrate_laguerre with use_mpmath parameter
    |-- A6: _integrate_jacobi with mpmath.quad fallback
    |-- A7: execute_quadrature dispatch propagation
    |-- A8: Convergence checking for mixed types (mpf vs float)
    |
    v  (optional, depends on Phase 2 being complete)
Phase 3: Optional Adaptive Fallback
    |-- Add _integrate_mpmath_adaptive method
    |-- Wire into execute_quadrature as last-resort fallback
```

---

## 6. SUMMARY OF ALL `float()` CASTS THAT MUST BE CONDITIONALIZED

### Analyzer file (`quadrature_analyzer_d_adapted_v2_G3.py`):

| Line | Current Code | Fix Required |
|------|-------------|--------------|
| 776 | `value_n = float(value)` | Conditional: skip when use_mpmath=True (Change A8) |
| 791 | `value_2n = float(value_2n)` | Conditional: skip when use_mpmath=True (Change A8) |
| 823 | `float(result) * scale` in `_integrate_legendre` | Return mp.mpf in mpmath path (Change A2) |
| 834 | `float(q.clenshaw_curtis_quadrature(...))` | Return mp.mpf in mpmath path (Change A3) |
| 836 | `float(clencurt_integrate_interval(...))` | Return mp.mpf in mpmath path (Change A3) |
| 873 | `float(quad.integrate(stripped))` in `_integrate_hermite` | Return mp.mpf in mpmath path (Change A4) |
| 1119 | `float(total) * weight_scale` in `_integrate_jacobi` | Return mp.mpf in mpmath path (Change A6) |
| 1193 | `float(quad.integrate(stripped))` in `_integrate_laguerre` | Return mp.mpf in mpmath path (Change A5) |

### Module-level files:

| File | Line(s) | Current Code | Fix Required |
|------|---------|-------------|--------------|
| `hermite/integration.py` | 58 | `[float(x) for x in n_sorted]` | Bypassed by new `integrate_mp` method (H1) |
| `legendre/integration.py` | 171 | `[float(v) for v in nodes_mp]` | Already handled by existing `HighPrecisionGaussLegendre` |
| `laguerre/integration.py` | 41-42 | `[float(vals[i]) ...]` | Bypassed by new `integrate_mp` method (G1) |

---

## 7. EXPECTED BEHAVIOR AFTER FIX

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

## 8. RISK ASSESSMENT AND MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Performance regression (mpmath is slow) | HIGH | MEDIUM | `use_mpmath` defaults to False; only activated explicitly by user |
| Breaking existing tests expecting float return types | MEDIUM | LOW | `QuadratureResult.value` remains float; internal mpf preserved until final cast in Change A8 |
| mpmath.quad for Jacobi may not converge for strong singularities | LOW | LOW | Alpha/beta < 1 ensures integrable singularities; fallback to numpy path available |
| Sympy `lambdify(modules='mpmath')` fails for some expressions | MEDIUM | MEDIUM | Add try/except with fallback: `lambda x: mp.mpf(f(float(x)))` |
| Fallback paths (Legendre on finite window) not mpmath-aware | LOW | MEDIUM | Change A4/A5 ensure recursive calls also use `use_mpmath=True` |

---

## 9. VERIFICATION STRATEGY

After implementation, verify with these test cases:

```python
# Test 1: Underflow case — exp(-100) should NOT be zero in mpmath path
analyzer.execute_quadrature("exp(-(x-5)**2)", interval=(0, 10), use_mpmath=True)
# The peak is at x=5 where exp(0)=1. At x=0: exp(-25) ≈ 1.4e-11 (fine in float64).
# But the stripped Hermite integrand involves exp(x^2)*exp(-(x-5)^2) = exp(10x - 25),
# which at large nodes overflows in float64 but is fine in mpmath.

# Test 2: Compare precision between use_mpmath=True and False
r_float = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=False)
r_mp    = analyzer.execute_quadrature("exp(-x**2) * cos(x)", use_mpmath=True)
# r_mp should agree with a known high-precision reference to >15 digits.

# Test 3: Infinite interval with algebraic decay (no Gaussian weight match)
analyzer.execute_quadrature("1 / (1 + x**4)", use_mpmath=True)
# This tests the Legendre finite-window fallback path with mpmath.
```

---

## 10. FILES TO MODIFY — COMPLETE INVENTORY

| # | File | Changes | Lines Affected |
|---|------|---------|---------------|
| 1 | `hermite/integration.py` | Add `integrate_mp` method | ~30 new lines |
| 2 | `laguerre/integration.py` | Add `integrate_mp` method | ~30 new lines |
| 3 | `legendre/integration.py` | Add `integrate_mp` bridge to `LegendreQuadrature` | ~15 new lines |
| 4 | `chebyshev/integration.py` | Add `integrate_on_interval_mp` bridge | ~15 new lines |
| 5 | `quadrature_analyzer_d_adapted_v2_G3.py` | A1-A8: Full analyzer refactoring | ~60 modified lines, ~40 new lines |

**Total estimated effort:** ~190 lines of code changes across 5 files.