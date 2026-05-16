# Quadrature Analyzer v2 G3 (mpmath) — User Guide

**File:** `api_new/quadrature_analyzer_d_adapted_v2_G3_mpmath.py`

This document provides a comprehensive guide to the **QuadratureAnalyzer**, an intelligent numerical integration engine that automatically selects the optimal orthogonal polynomial family and quadrature method for any given integrand. It leverages the four-layer HP-Z4 architecture of the `chebyshev`, `hermite`, `laguerre`, and `legendre` modules, with full **mpmath arbitrary-precision** support across all families.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture & Design](#2-architecture--design)
3. [Core Data Types](#3-core-data-types)
4. [The Analysis Pipeline (`analyze()`)](#4-the-analysis-pipeline-analyze)
5. [The Execution Pipeline (`execute_quadrature()`)](#5-the-execution-pipeline-execute_quadrature)
6. [Module Integration Map](#6-module-integration-map)
7. [NumPy vs mpmath Modes](#7-numpy-vs-mpmath-modes)
8. [Category Fixes (A–G)](#8-category-fixes-ag)
9. [Complete Usage Examples](#9-complete-usage-examples)
10. [Reference: All Public Methods](#10-reference-all-public-methods)

---

## 1. Overview

The QuadratureAnalyzer solves the problem of **automatic numerical integration** by combining two phases:

| Phase | Method | Purpose |
|-------|--------|---------|
| **Analysis** | `analyze()` | Inspects the integrand symbolically to classify its properties and recommend the best polynomial family |
| **Execution** | `execute_quadrature()` | Performs the actual integration with automatic convergence checking, using either NumPy (float64) or mpmath (arbitrary precision) |

### What it does automatically:

1. **Interval inference** — detects whether the integral is over a finite, semi-infinite, or infinite domain
2. **Singularity detection** — finds poles, log singularities, and square-root branch points
3. **Derivative growth probing** — classifies smoothness (bounded → polynomial → exponential → super-exponential)
4. **Decay analysis** — identifies Gaussian, exponential, algebraic, or oscillatory decay on infinite domains
5. **Periodicity detection** — recognizes trigonometric periodicity
6. **Family recommendation** — selects Legendre, Chebyshev, Hermite, or Laguerre based on all above factors
7. **Degree selection** — suggests optimal node counts (n) with oscillation-aware bumping
8. **Convergence verification** — compares n vs 2n results to confirm tolerance is met

---

## 2. Architecture & Design

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    QuadratureAnalyzer                        │
│                                                              │
│  ┌──────────┐    FunctionAnalysis     ┌──────────────────┐  │
│  │ analyze() │ ──────────────────────► │ execute_quad()   │  │
│  │          │                          │                  │  │
│  │ • Parse  │   Properties detected:   │ Dispatches to:    │  │
│  │ • Infer  │   - Interval type        │ • _integrate_     │  │
│  │ • Find   │   - Singularities        │   legendre()      │  │
│  │ • Probe  │   - Derivative growth    │ • _integrate_     │  │
│  │ • Recommend│- Decay type           │   chebyshev()     │  │
│  └──────────┘   - Periodicity          │ • _integrate_     │  │
│                                         │   hermite()       │  │
│                                         │ • _integrate_     │  │
│                                         │   laguerre()      │  │
│                                         └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                                                    │
         ▼                                                    ▼
┌──────────────────────┐                          ┌──────────────────────┐
│  Symbolic Analysis   │                          │  Quadrature Modules  │
│  (SymPy)             │                          │                      │
│  • parse_expr()      │                          │  legendre module     │
│  • diff(), solve()   │                          │    LegendreQuadrature│
│  • as_numer_denom()  │                          │    .integrate_mp()   │
└──────────────────────┘                          │                      │
                                                  │  chebyshev module    │
                                                  │    ClenshawCurtisMP  │
                                                  │    clencurt_integrate│
                                                  │                      │
                                                  │  hermite module      │
                                                  │    GaussHermiteQuad. │
                                                  │    .integrate_mp()   │
                                                  │                      │
                                                  │  laguerre module     │
                                                  │    LaguerreQuadrature│
                                                  │    .integrate_mp()   │
                                                  └──────────────────────┘
```

### Recommendation Decision Tree

```
Interval type?
│
├─ infinite (-∞, +∞)
│  ├─ Gaussian decay (e^(-cx²), c≥1) → HERMITE (high confidence)
│  ├─ Exponential decay              → HERMITE (medium)
│  └─ Other/slow decay               → LEGENDRE with finite window (G3 fallback)
│
├─ semi-infinite [0, +∞)
│  ├─ Exponential decay              → LAGUERRE (high)
│  ├─ Algebraic decay                → LAGUERRE (medium)
│  └─ Other                          → LAGUERRE (default for semi-infinite)
│
└─ finite [a, b]
   ├─ Endpoint singularity           → CHEBYSHEV (high) or Gauss-Jacobi (Category C)
   ├─ Interior singularity           → LEGENDRE (low confidence — split recommended)
   ├─ Periodic                       → CHEBYSHEV (high)
   └─ Smooth, no singularities       → LEGENDRE (high)
```

---

## 3. Core Data Types

### `PolynomialFamily` (Enum)

```python
from quadrature_analyzer_d_adapted_v2_G3_mpmath import PolynomialFamily

PolynomialFamily.LEGENDRE    # "Legendre"
PolynomialFamily.CHEBYSHEV   # "Chebyshev"
PolynomialFamily.HERMITE     # "Hermite"
PolynomialFamily.LAGUERRE    # "Laguerre"
```

### `FunctionAnalysis` (dataclass)

Returned by `analyze()`. Contains all detected properties:

| Field | Type | Description |
|-------|------|-------------|
| `original_expr` | str | The original expression string |
| `sympy_expr` | sympy.Expr | Parsed SymPy expression |
| `variable` | str | Integration variable (default "x") |
| `interval_type` | str | `"finite"`, `"semi_infinite"`, or `"infinite"` |
| `interval_a`, `interval_b` | float | Interval endpoints |
| `singularities` | list[dict] | Detected singularities with location and kind |
| `has_endpoint_singularity` | bool | True if singularity at a or b |
| `has_interior_singularity` | bool | True if singularity in (a, b) |
| `derivative_growth_rate` | str | `"bounded"`, `"polynomial"`, `"exponential"`, `"super_exponential"` |
| `decay_type` | str | `"none"`, `"gaussian"`, `"exponential"`, `"algebraic"`, `"oscillatory"` |
| `decay_rate` | float | Rate parameter (e.g., c in e^(-cx²)) |
| `hermite_compatible` | bool | True if Gaussian weight matches standard Hermite |
| `is_periodic_on_interval` | bool | Detected periodicity |
| `approximate_period` | Optional[float] | Estimated period |
| `recommended_family` | PolynomialFamily | Recommended quadrature family |
| `confidence` | str | `"high"`, `"medium"`, or `"low"` |
| `recommendation_reason` | str | Human-readable explanation |
| `suggested_min_n`, `suggested_max_n` | int | Suggested node count range |
| `degree_criteria` | list[str] | Detailed criteria explanations |
| `left_singularity_alpha` | float | Left endpoint singularity exponent (Category C) |
| `right_singularity_beta` | float | Right endpoint singularity exponent (Category C) |
| `has_algebraic_endpoint_singularity` | bool | True if Jacobi weight matching needed |

### `QuadratureResult` (dataclass)

Returned by `execute_quadrature()`:

| Field | Type | Description |
|-------|------|-------------|
| `value` | float | Computed integral value |
| `family_used` | PolynomialFamily | Family actually used for integration |
| `n_nodes` | int | Number of quadrature nodes used |
| `converged` | bool | True if n vs 2n comparison met tolerance |
| `error_estimate` | Optional[float] | Absolute error estimate from convergence check |
| `message` | str | Status message ("Converged" or diagnostic) |

---

## 4. The Analysis Pipeline (`analyze()`)

### Signature

```python
analysis = analyzer.analyze(
    expression,        # str or sympy.Expr
    interval=None,     # Optional tuple (a, b); auto-inferred if None
    *,
    variable="x",      # Integration variable name
    max_deriv_order=6  # Maximum derivative order to probe
)
```

### Step-by-Step Process

#### Step 1: Parse Expression

Accepts either a string or SymPy expression. Strings are parsed with implicit multiplication support:

```python
# Both work:
analyzer.analyze("exp(-x**2)")
analyzer.analyze(sp.exp(-sp.Symbol('x')**2))
```

#### Step 2: Infer Interval (if not provided)

The analyzer inspects the expression structure to guess the natural domain:

| Expression Pattern | Inferred Interval | Reasoning |
|--------------------|-------------------|-----------|
| `exp(-c·x²)` where c > 0 | (-∞, +∞) | Gaussian decay on infinite domain |
| `exp(-c·x)` where c > 0 | [0, +∞) | Exponential decay on semi-infinite domain |
| `log(x)` | [0, +∞) | Logarithm implies x ≥ 0 |
| Rational with negative powers of x | [0, +∞) | Algebraic singularity at origin |
| Everything else | [-1, 1] | Default finite interval |

#### Step 3: Find Singularities

Scans for four types:

```python
# Pole detection (denominator roots)
"1 / (x - 2)"           # → pole at x=2

# Logarithmic singularities (log argument = 0)
"log(x)"                # → log_zero at x=0

# Square-root branch points
"sqrt(x + 1)"          # → sqrt_zero at x=-1

# Tangent poles
"tan(x)"               # → tan_pole at ±π/2, ±3π/2, ...
```

#### Step 4: Probe Derivative Growth

Evaluates derivatives up to `max_deriv_order` at Gauss-Legendre probe points and computes the log-log growth ratio:

| Classification | Ratio Range | Implication |
|----------------|-------------|-------------|
| `"bounded"` | < 1.5 | Exponential convergence; n=8–32 suffices |
| `"polynomial"` | 1.5–2.5 | Algebraic convergence O(n⁻ᵖ); use n=32–128 |
| `"exponential"` | 2.5–6.0 | Slow algebraic convergence; use n=64–256 |
| `"super_exponential"` | ≥ 6.0 | Consider splitting the interval |

#### Step 5: Probe Decay (infinite/semi-infinite domains)

```python
# Gaussian decay detection
"exp(-x**2)"    # → ("gaussian", c=1.0, hermite_compatible=True)
"exp(-3*x**2)"  # → ("gaussian", c=3.0, hermite_compatible=True)

# Exponential decay detection
"exp(-2*x)"     # → ("exponential", rate=2.0, hermite_compatible=False)

# Algebraic decay detection
"1/x**3"        # → ("algebraic", power=3.0, hermite_compatible=False)

# Oscillatory detection (trig functions without decay)
"sin(x)/x"      # → ("oscillatory", 0.0, False)
```

#### Step 6: Probe Periodicity

Detects trigonometric atoms and computes the fundamental period from coefficient extraction:

```python
"cos(3*x)"       # → periodic=True, period≈2.094 (2π/3)
"sin(x) + cos(2x)" # → periodic=True, period≈6.283 (LCM of periods)
```

#### Step 7: Recommend Family and Degree

Combines all detected properties into a recommendation with confidence level and suggested node count range.

### Example Analysis Output

```python
from quadrature_analyzer_d_adapted_v2_G3_mpmath import QuadratureAnalyzer

analyzer = QuadratureAnalyzer()

analysis = analyzer.analyze("exp(-x**2) * sin(x)", interval=(-1, 1))

print(f"Interval type:    {analysis.interval_type}")
# → Interval type:    finite

print(f"Singularities:    {analysis.singularities}")
# → Singularities:    []

print(f"Growth rate:      {analysis.derivative_growth_rate}")
# → Growth rate:      bounded

print(f"Periodic:         {analysis.is_periodic_on_interval}")
# → Periodic:         True (period ~6.283)

print(f"Recommended:      {analysis.recommended_family.value}")
# → Recommended:      Chebyshev

print(f"Confidence:       {analysis.confidence}")
# → Confidence:       high

print(f"Suggested n:      [{analysis.suggested_min_n}, {analysis.suggested_max_n}]")
# → Suggested n:      [8, 32]
```

---

## 5. The Execution Pipeline (`execute_quadrature()`)

### Signature

```python
result = analyzer.execute_quadrature(
    expression,        # str or sympy.Expr
    interval=None,     # Optional tuple (a, b)
    *,
    variable="x",      # Integration variable
    n=None,            # Node count (auto-selected from analysis if None)
    tol=1e-12,         # Convergence tolerance
    use_mpmath=False   # True for arbitrary-precision mpmath pipeline
)
```

### Execution Flow

```
execute_quadrature()
│
├─ 1. Run analyze() to get FunctionAnalysis
│
├─ 2. Category C: Check endpoint singularities → Gauss-Jacobi if needed
│     └─ _extract_endpoint_singularity_exponents()
│        └─ Log-log slope analysis at endpoints
│
├─ 3. G3 Fallback: Infinite windowing for Legendre on infinite domains
│     └─ _compute_effective_support() → finite [−L, +L]
│
├─ 4. Category B: Oscillation bump — inflate n if high-frequency detected
│     └─ _estimate_max_oscillation_frequency()
│        └─ _compute_oscillation_safe_n()
│
├─ 5. Dispatch to family-specific integrator (with use_mpmath propagation)
│     ├─ Legendre:    _integrate_legendre()      → legendre.LegendreQuadrature
│     ├─ Chebyshev:   _integrate_chebyshev()     → chebyshev.ClenshawCurtisMP / clencurt_integrate_interval
│     ├─ Hermite:     _integrate_hermite()       → hermite.GaussHermiteQuadrature
│     └─ Laguerre:    _integrate_laguerre()      → laguerre.LaguerreQuadrature
│
├─ 6. Convergence check (n vs 2n comparison)
│     ├─ Absolute error: |I(2n) − I(n)| < tol
│     └─ Relative error: |I(2n) − I(n)| / |I(n)| < tol
│
└─ 7. Return QuadratureResult
```

### Convergence Checking Detail

The analyzer computes the integral at both `n` and `2n` nodes, then checks:

```python
err = abs(value_2n - value_n)
rel_err = err / (abs(value_n) + 1e-30)   # avoid division by zero
converged = (err < tol) or (rel_err < tol)
```

In **mpmath mode**, this comparison is done in full arbitrary precision before casting to float:

```python
# mpmath path — all arithmetic in mp.mpf
err_mp = abs(value_2n_mp - value_n_mp)
rel_err_mp = err_mp / (abs(value_n_mp) + mp.mpf('1e-30'))
tol_mp = mp.mpf(str(tol))
converged = bool(err_mp < tol_mp or rel_err_mp < tol_mp)
```

---

## 6. Module Integration Map

This section documents exactly how the analyzer uses each polynomial module, with both NumPy and mpmath paths.

### 6.1 Legendre Module (`legendre`)

**Import:** `from legendre import LegendreQuadrature`

| Analyzer Method | NumPy Path | mpmath Path |
|-----------------|------------|-------------|
| `_integrate_legendre()` | `LegendreQuadrature(n, use_mpmath=False).integrate(transformed)` × scale | `LegendreQuadrature(n, use_mpmath=True, dps=80).integrate_mp(expr, a=a, b=b)` |

**What it does:**
- **NumPy path**: Maps [a,b] → [-1,1], calls standard float64 integration, multiplies by Jacobian `(b-a)/2`
- **mpmath path**: Calls `LegendreQuadrature.integrate_mp()` which internally uses `HighPrecisionGaussLegendre` with 80 dps and handles the [a,b] interval transformation in mpmath arithmetic

**Fallback:** When Legendre is recommended for an infinite domain (G3 fallback), the analyzer computes a finite window `[−L, +L]` using `_compute_effective_support()` and then calls `_integrate_legendre()` on that window.

### 6.2 Chebyshev Module (`chebyshev`)

**Imports:**
- NumPy: `from chebyshev import clencurt_integrate_interval, ChebyshevQuadrature`
- mpmath: `from chebyshev.integration_mp import ClenshawCurtisMP`

| Analyzer Method | NumPy Path | mpmath Path |
|-----------------|------------|-------------|
| `_integrate_chebyshev()` | `clencurt_integrate_interval(func, a, b, n)` or `ChebyshevQuadrature().clenshaw_curtis_quadrature(stripped, n)` for endpoint singularities | `ClenshawCurtisMP(n, dps=80).integrate_on_interval(f_mp, a, b)` |

**What it does:**
- **NumPy path (smooth)**: Direct Clenshaw-Curtis on [a,b] via `clencurt_integrate_interval`
- **NumPy path (endpoint singularity)**: Strips the Chebyshev weight `1/√(1−x²)` from the integrand, then uses `ChebyshevQuadrature.clenshaw_curtis_quadrature()` on [-1,1]
- **mpmath path**: Uses `ClenshawCurtisMP` with 80 dps; lambdifies the SymPy expression with mpmath modules; calls `integrate_on_interval(f_mp, a, b)` which handles interval mapping in full precision

### 6.3 Hermite Module (`hermite`)

**Import:** `from hermite import GaussHermiteQuadrature`

| Analyzer Method | NumPy Path | mpmath Path |
|-----------------|------------|-------------|
| `_integrate_hermite()` | Strips e^(-x²) weight, calls `GaussHermiteQuadrature(n).integrate(stripped)` | Strips e^(-x²) weight symbolically, calls `GaussHermiteQuadrature(n, use_mpmath=True, dps=80).integrate_mp(stripped_expr, dps=80)` |

**What it does:**
- **Both paths**: Divides the integrand by e^(-x²) to get the "stripped" function that Gauss-Hermite quadrature expects (since the weight is built into the nodes/weights)
- **Overflow protection**: Tests the stripped function at x = 3, 5, 7. If values exceed thresholds, falls back to finite-window Legendre integration
- **mpmath overflow**: If `|result_mp| > 1e30`, falls back to `_integrate_legendre()` with mpmath on a finite window

### 6.4 Laguerre Module (`laguerre`)

**Import:** `from laguerre import LaguerreQuadrature`

| Analyzer Method | NumPy Path | mpmath Path |
|-----------------|------------|-------------|
| `_integrate_laguerre()` | Strips e^(-x) weight, calls `LaguerreQuadrature(n, alpha=0.0).integrate(stripped)` | Strips e^(-x) weight symbolically, calls `LaguerreQuadrature(n, alpha=0.0, use_mpmath=True).integrate_mp(stripped_expr, dps=80)` |

**What it does:**
- **Both paths**: Divides the integrand by e^(-x) to get the "stripped" function (since Gauss-Laguerre weight is built in)
- **Multi-level fallback chain** (NumPy path only):
  1. Tests stripped function at x = 10, 30, 60 for overflow → fall back to Legendre on [0, L]
  2. Checks growth ratio between test points > 1e6 → fall back to Legendre
  3. Detects oscillations + growth simultaneously → fall back to Legendre
  4. Non-finite result → fall back to Legendre
- **mpmath overflow**: If `|result_mp| > 1e30`, falls back to `_integrate_legendre()` with mpmath on [0, L]

### 6.5 Jacobi Weight Integration (Category C)

**Import:** `from scipy.special import roots_jacobi` (NumPy path), `mpmath.quad` (mpmath path)

| Analyzer Method | NumPy Path | mpmath Path |
|-----------------|------------|-------------|
| `_integrate_with_jacobi_weight()` | Gauss-Jacobi quadrature via scipy.special.roots_jacobi | `mpmath.quad()` with explicit Jacobi weight `(b-x)^α · (x-a)^β` |

**When triggered:** When endpoint singularities are detected on a finite interval and the singularity exponents α, β satisfy 0 < α, β < 1.

---

## 7. NumPy vs mpmath Modes

### The `use_mpmath` Flag

The single boolean parameter `use_mpmath` in `execute_quadrature()` controls the entire precision pipeline:

```python
# ── Standard mode (NumPy float64) ───────────────────────
result = analyzer.execute_quadrature(
    "exp(-x**2)", interval=(-1, 1), use_mpmath=False
)
# → ~15-17 significant digits, fast execution

# ── High-precision mode (mpmath arbitrary precision) ─────
result = analyzer.execute_quadrature(
    "exp(-x**2)", interval=(-1, 1), use_mpmath=True
)
# → 80 decimal places of internal precision, slower but more accurate
```

### What Changes Between Modes

| Aspect | `use_mpmath=False` | `use_mpmath=True` |
|--------|-------------------|-------------------|
| **Lambdification** | `lambdify(..., modules="numpy")` | `lambdify(..., modules='mpmath')` |
| **Legendre integration** | `LegendreQuadrature.integrate()` (float64) | `LegendreQuadrature.integrate_mp()` → mp.mpf |
| **Chebyshev integration** | `clencurt_integrate_interval()` (NumPy arrays) | `ClenshawCurtisMP.integrate_on_interval()` → mp.mpf |
| **Hermite integration** | `GaussHermiteQuadrature.integrate()` (float64) | `GaussHermiteQuadrature.integrate_mp(dps=80)` → mp.mpf |
| **Laguerre integration** | `LaguerreQuadrature.integrate()` (float64) | `LaguerreQuadrature.integrate_mp(dps=80)` → mp.mpf |
| **Convergence check** | Standard float arithmetic | Full mp.mpf arithmetic, then cast to float at the end |
| **2n cap** | Up to 1000 nodes | Capped at 150 (Golub-Welsch slow in mpmath) |

### When to Use mpmath Mode

```python
# Case 1: High-accuracy verification
result_np = analyzer.execute_quadrature("sin(x)/x", interval=(0, 1), use_mpmath=False)
result_mp = analyzer.execute_quadrature("sin(x)/x", interval=(0, 1), use_mpmath=True)
print(f"NumPy:  {result_np.value:.16f}")
print(f"mpmath: {result_mp.value:.16f}")

# Case 2: Near-singular integrands where float64 loses digits
result = analyzer.execute_quadrature(
    "1 / sqrt(x)", interval=(0, 1), use_mpmath=True
)

# Case 3: Very tight tolerance requirements
result = analyzer.execute_quadrature(
    "exp(-x**2) * cos(50*x)", interval=(-1, 1), tol=1e-20, use_mpmath=True
)
```

---

## 8. Category Fixes (A–G)

The mpmath version implements a series of fixes labeled A1–A8:

| Fix | Description | Implementation |
|-----|-------------|----------------|
| **A1** | Module-level mpmath import | `import mpmath as mp` at top of file |
| **A2** | Legendre mpmath integration | `_integrate_legendre()` accepts `use_mpmath`; calls `LegendreQuadrature.integrate_mp()` |
| **A3** | Chebyshev mpmath integration | `_integrate_chebyshev()` uses `ClenshawCurtisMP` when `use_mpmath=True` |
| **A4** | Hermite full mpmath pipeline | `_integrate_hermite()` strips weight symbolically, calls `GaussHermiteQuadrature.integrate_mp()`, returns mp.mpf; overflow fallback also uses mpmath Legendre |
| **A5** | Laguerre mpmath integration | `_integrate_laguerre()` calls `LaguerreQuadrature.integrate_mp()`; overflow fallback uses mpmath Legendre |
| **A6** | Jacobi weight mpmath support | `_integrate_jacobi()` and `_integrate_with_jacobi_weight()` use `mpmath.quad` when `use_mpmath=True` |
| **A7** | Dispatch propagation | `execute_quadrature()` propagates `use_mpmath` to ALL integration methods including convergence check at 2n |
| **A8** | Convergence in mp.mpf | When `use_mpmath=True`, convergence comparison uses full mp.mpf arithmetic; only casts to float for the final `QuadratureResult` |

### Category B: Oscillation Handling

Detects high-frequency oscillations via `_estimate_max_oscillation_frequency()` and inflates the node count:

```python
# For sin(50*x) on [-1, 1]:
omega_max = 50.0          # max frequency from derivative of phase
n_half_periods = 50 * 2 / π ≈ 31.8
min_osc_nodes = 10 × 31.8 ≈ 318    # at least 10 nodes per half-period
```

### Category C: Endpoint Singularity Matching

Extracts singularity exponents via log-log slope analysis and uses Gauss-Jacobi quadrature with matching weight `(b-x)^α · (x-a)^β`:

```python
# For 1/sqrt(x) on [0, 1]:
left_alpha ≈ 0.5    # detected from log-log slope near x=0
right_beta = 0.0    # no singularity at right endpoint
# → Uses Jacobi weight (1-x)^0 · (x-0)^0.5
```

### Category D: Overflow Protection

Each integrator has overflow detection with automatic fallback to finite-window Legendre integration when the stripped integrand grows too large.

---

## 9. Complete Usage Examples

### Example 1: Basic Analysis + Execution

```python
from quadrature_analyzer_d_adapted_v2_G3_mpmath import QuadratureAnalyzer

analyzer = QuadratureAnalyzer()

# ── Analyze first, then execute ────────────────────────
analysis = analyzer.analyze("exp(-x**2) * sin(x)", interval=(-1, 1))

print(f"Recommended family: {analysis.recommended_family.value}")
print(f"Confidence:         {analysis.confidence}")
print(f"Suggested n range:  [{analysis.suggested_min_n}, {analysis.suggested_max_n}]")

# Execute with default settings (NumPy float64)
result = analyzer.execute_quadrature("exp(-x**2) * sin(x)", interval=(-1, 1))
print(f"Result:     {result.value:.15e}")
print(f"Converged:  {result.converged}")
print(f"Error est:  {result.error_estimate:.3e}")

# Execute with mpmath (arbitrary precision)
result_mp = analyzer.execute_quadrature(
    "exp(-x**2) * sin(x)", interval=(-1, 1), use_mpmath=True
)
print(f"mpmath result: {result_mp.value:.15e}")
```

### Example 2: Auto-Interval Inference (Infinite Domain)

```python
# No interval specified — analyzer detects Gaussian decay → (-∞, +∞)
result = analyzer.execute_quadrature("exp(-x**2)")
print(f"∫_{-∞}^{+∞} e^(-x²) dx = {result.value:.15f}")  # ≈ √π ≈ 1.7724538509...
print(f"Family: {result.family_used.value}")              # Hermite

# With mpmath for higher precision
result_mp = analyzer.execute_quadrature("exp(-x**2)", use_mpmath=True)
```

### Example 3: Semi-Infinite Domain (Laguerre)

```python
# log(x)·e^(-x) on [0, +∞) — auto-detected from log(x) pattern
result = analyzer.execute_quadrature("log(x) * exp(-x)")
print(f"Result: {result.value:.15f}")  # ≈ -γ (Euler-Mascheroni constant)

# With mpmath
result_mp = analyzer.execute_quadrature("log(x) * exp(-x)", use_mpmath=True)
```

### Example 4: Endpoint Singularity (Chebyshev / Jacobi)

```python
# 1/√(1-x²) on [-1, 1] — endpoint singularities at both ends
result = analyzer.execute_quadrature("1 / sqrt(1 - x**2)", interval=(-1, 1))
print(f"Result: {result.value:.15f}")  # ≈ π

# With mpmath (handles singularities better)
result_mp = analyzer.execute_quadrature(
    "1 / sqrt(1 - x**2)", interval=(-1, 1), use_mpmath=True
)
```

### Example 5: Custom Node Count and Tolerance

```python
# Force specific node count
result = analyzer.execute_quadrature(
    "exp(-x**2)", interval=(-1, 1), n=64, tol=1e-14
)

# Very tight tolerance with mpmath
result_mp = analyzer.execute_quadrature(
    "cos(3*x)", interval=(-float('pi'), float('pi')),
    tol=1e-20, use_mpmath=True
)
```

### Example 6: SymPy Expression Input

```python
import sympy as sp

x = sp.Symbol('x')
expr = sp.exp(-x**2) * sp.sin(x)

# Pass SymPy expression directly (no string parsing needed)
result = analyzer.execute_quadrature(expr, interval=(-1, 1), variable="x")
print(f"Result: {result.value:.15f}")

# mpmath mode works seamlessly with SymPy expressions
result_mp = analyzer.execute_quadrature(
    expr, interval=(-1, 1), variable="x", use_mpmath=True
)
```

### Example 7: NumPy vs mpmath Comparison

```python
import numpy as np
import mpmath as mp

test_cases = [
    ("exp(-x**2)", (-1, 1), "Gaussian on finite interval"),
    ("exp(-x**2)", None, "Gaussian on infinite domain (Hermite)"),
    ("log(x) * exp(-x)", None, "Log × exponential decay (Laguerre)"),
    ("1 / sqrt(1 - x**2)", (-1, 1), "Endpoint singularity (Chebyshev/Jacobi)"),
]

for expr_str, interval, desc in test_cases:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Expression: {expr_str}")

    # NumPy mode
    r_np = analyzer.execute_quadrature(expr_str, interval=interval, use_mpmath=False)

    # mpmath mode
    r_mp = analyzer.execute_quadrature(expr_str, interval=interval, use_mpmath=True)

    print(f"  NumPy:   {r_np.value:.16e}  (n={r_np.n_nodes}, converged={r_np.converged})")
    print(f"  mpmath:  {r_mp.value:.16e}  (n={r_mp.n_nodes}, converged={r_mp.converged})")

    if r_np.converged and r_mp.converged:
        diff = abs(r_np.value - r_mp.value)
        print(f"  Difference: {diff:.2e}")
```

### Example 8: Using the Built-in Demo

The file includes a self-test in `__main__`:

```bash
python api_new/quadrature_analyzer_d_adapted_v2_G3_mpmath.py
```

This runs through seven representative examples covering all four polynomial families and both finite/infinite domains.

---

## 10. Reference: All Public Methods

### `QuadratureAnalyzer` Class

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__()` | `(default_variable="x")` | Create analyzer instance |
| `analyze()` | `(expression, interval=None, *, variable="x", max_deriv_order=6)` → `FunctionAnalysis` | Full symbolic analysis of integrand |
| `execute_quadrature()` | `(expression, interval=None, *, variable="x", n=None, tol=1e-12, use_mpmath=False)` → `QuadratureResult` | Execute integration with convergence check |
| `recommend_usage()` | `(analysis: FunctionAnalysis)` → `str` | Generate usage snippet for recommended family |

### Internal Methods (Documented for Understanding)

| Method | Description | Module Used |
|--------|-------------|-------------|
| `_integrate_legendre()` | Gauss-Legendre on [a,b] with interval mapping | `legendre.LegendreQuadrature` |
| `_integrate_chebyshev()` | Clenshaw-Curtis on [a,b], handles endpoint singularities | `chebyshev.clencurt_integrate_interval`, `chebyshev.integration_mp.ClenshawCurtisMP` |
| `_integrate_hermite()` | Gauss-Hermite on (-∞,+∞), strips e^(-x²) weight | `hermite.GaussHermiteQuadrature` |
| `_integrate_laguerre()` | Gauss-Laguerre on [0,+∞), strips e^(-x) weight, multi-level fallback | `laguerre.LaguerreQuadrature` |
| `_integrate_jacobi()` | Jacobi-weighted quadrature for endpoint singularities | `scipy.special.roots_jacobi` / `mpmath.quad` |
| `_integrate_with_jacobi_weight()` | Full Jacobi pipeline: strips singularity, calls _integrate_jacobi | — |
| `_compute_effective_support()` | Compute finite window L for infinite-domain Legendre fallback | — |
| `_estimate_max_oscillation_frequency()` | Category B: detect max oscillation frequency from trig atoms | — |
| `_compute_oscillation_safe_n()` | Category B: inflate n based on oscillation count | — |
| `_extract_endpoint_singularity_exponents()` | Category C: log-log slope analysis at endpoints | — |

---

## Appendix A: Quick Start

```python
# Minimal working example
from quadrature_analyzer_d_adapted_v2_G3_mpmath import QuadratureAnalyzer

analyzer = QuadratureAnalyzer()

# One-liner: analyze and integrate automatically
result = analyzer.execute_quadrature("exp(-x**2)")
print(f"∫ e^(-x²) dx = {result.value:.15f}  (family={result.family_used.value}, n={result.n_nodes})")
```

## Appendix B: Module Dependency Summary

The analyzer depends on these modules from the HP-Z4 architecture:

| Analyzer Component | Depends On | Layer Used |
|--------------------|------------|------------|
| `_integrate_legendre()` (NumPy) | `legendre.LegendreQuadrature` | L4 (integration.py) |
| `_integrate_legendre()` (mpmath) | `legendre.LegendreQuadrature.integrate_mp()` → `HighPrecisionGaussLegendre` | L4 + mpmath |
| `_integrate_chebyshev()` (NumPy) | `chebyshev.clencurt_integrate_interval`, `ChebyshevQuadrature` | L4 (integration.py) |
| `_integrate_chebyshev()` (mpmath) | `chebyshev.integration_mp.ClenshawCurtisMP` | L4 + mpmath |
| `_integrate_hermite()` (NumPy) | `hermite.GaussHermiteQuadrature.integrate()` | L4 (integration.py) |
| `_integrate_hermite()` (mpmath) | `hermite.GaussHermiteQuadrature.integrate_mp()` | L4 + mpmath |
| `_integrate_laguerre()` (NumPy) | `laguerre.LaguerreQuadrature.integrate()` | L4 (integration.py) |
| `_integrate_laguerre()` (mpmath) | `laguerre.LaguerreQuadrature.integrate_mp()` | L4 + mpmath |

All four modules are used exclusively through their **Layer 4 (integration)** interfaces, which internally orchestrate the lower layers (symbolic, high-precision, numerical). The analyzer never directly imports from Layer 1–3.