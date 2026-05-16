# Plan: Add Multi-Precision (MP) Tests for Hermite, Laguerre, Legendre

## 1. Current State Analysis

### Chebyshev — Reference Pattern (ALREADY DONE)
- **Infrastructure**: Dedicated `chebyshev/integration_mp.py` module with full mpmath classes and convenience functions
- **Tests**: `chebyshev_tests/test_chebyshev_06_integration_mp.py` — 10 comprehensive test sections:
  - TEST 1: Gauss-Chebyshev Type I (∫ dx/sqrt(1-x²) = π)
  - TEST 2: Gauss-Chebyshev Type II (∫ sqrt(1-x²) dx = π/2)
  - TEST 3: Clenshaw-Curtis MP polynomial integrals on [-1, 1]
  - TEST 4: Clenshaw-Curtis MP on arbitrary interval [a, b]
  - TEST 5: Chebyshev Projection / Approximation round-trip
  - TEST 6: Precision convergence (dps=50 vs dps=200)
  - TEST 7: chebyshev_transform_mp / inverse_chebyshev_transform_mp
  - TEST 8: get_nodes_weights_mp + map_nodes_to_interval_mp
  - TEST 9: Float64 bridge functions
  - TEST 10: gauss_chebyshev_mp convenience function

### Hermite — MP Tests MISSING
- **Infrastructure**: `hermite/integration.py` has built-in mpmath support (no separate integration_mp.py)
  - `GaussHermiteQuadrature(n, use_mpmath=True, dps=80)` 
  - `.integrate_mp(f_or_expr, dps)` — full mpmath pipeline returning mp.mpf
- **Existing non-MP tests**: test_hermite_01 (symbolic), 02 (high_precision), 03 (numerical), 04 (integration float64)

### Laguerre — MP Tests MISSING
- **Infrastructure**: `laguerre/integration.py` has built-in mpmath support (no separate integration_mp.py)
  - `LaguerreQuadrature(n, alpha, use_mpmath=True)` 
  - `.integrate_mp(f_or_expr, dps=80)` — full mpmath pipeline returning mp.mpf
- **Existing non-MP tests**: test_laguerre_01 (symbolic), 02 (high_precision), 03 (numerical), 04 (integration float64)

### Legendre — Partial MP Coverage (needs expansion)
- **Infrastructure**: `legendre/integration.py` has extensive mpmath support:
  - `LegendreQuadrature(n, use_mpmath=True, dps=80)` with `.integrate_mp(f_or_expr, a, b)` 
  - `HighPrecisionGaussLegendre(n, dps=80)` — true arbitrary precision (nodes/weights as mp.mpf)
  - `gauss_legendre_high_precision(n, dps)` factory function
- **Existing tests**: test_legendre_05 (precision comparison), test_legendre_06 (HighPrecisionGaussLegendre)
- **Gap**: No dedicated MP test file following the chebyshev pattern with PASS/FAIL assertions

---

## 2. Design Decision: Two Approaches for MP Infrastructure

The existing code uses two different patterns:

| Family | Pattern | Description |
|--------|---------|-------------|
| Chebyshev | Separate `integration_mp.py` | Dedicated module, clean separation |
| Hermite/Laguerre/Legendre | Built-in mpmath in `integration.py` | Single module with `use_mpmath` flag + `integrate_mp()` method |

**Recommendation**: For the MP tests, follow the existing pattern of each family (i.e., test what already exists). Do NOT create new integration_mp.py files for hermite/laguerre/legendre unless explicitly requested. The tests should exercise the mpmath capabilities that are already present in `integration.py`.

---

## 3. Plan: `hermite_tests/test_hermite_05_integration_mp.py`

Create a comprehensive MP test file following the chebyshev pattern (PASS/FAIL assertions, structured sections).

```
=== TEST 1: Gauss-Hermite Quadrature — ∫_{-∞}^{∞} e^{-x²/4} · e^{-x²} dx = sqrt(4π/5) ===
  - Use GaussHermiteQuadrature(n, use_mpmath=True, dps=80)
  - Verify convergence for n in [8, 16, 32]
  - Assert: n=32 result within tolerance of exact value

=== TEST 2: integrate_mp() — full mpmath pipeline ===
  - Use .integrate_mp(lambda x: mp.exp(-x**2 / 4), dps=80)
  - Verify return type is mp.mpf (not float)
  - Assert result within tolerance of sqrt(4π/5)

=== TEST 3: Gauss-Hermite — ∫_{-∞}^{∞} e^{-x²} · e^{-x²} dx = sqrt(π) ===
  - f(x) = 1 (weight is built-in), integral of weight function = sqrt(π)
  - Verify for n in [8, 16, 32]

=== TEST 4: Gauss-Hermite — polynomial integrands ===
  - ∫ x² · e^{-x²} dx = sqrt(π)/2 (odd powers should be ~0 by symmetry)
  - Test even and odd powers k in [0, 1, 2, 3, 4]

=== TEST 5: Hermite Projection round-trip ===
  - Use HermiteProjection(max_degree=16, use_mpmath=True)
  - Project f(x) = exp(-x²/2), reconstruct at test points
  - Assert max reconstruction error < tolerance

=== TEST 6: Precision convergence (dps=50 vs dps=200) ===
  - Integrate a challenging function using integrate_mp() with varying dps
  - Compare against mpmath quad reference at very high precision
  - Assert that higher dps yields more correct digits

=== TEST 7: Nodes and weights properties ===
  - Verify nodes are sorted ascending
  - Verify all weights positive
  - Verify sum of weights ≈ sqrt(π) (integral of weight function with f=1)

=== TEST 8: hermite_transform / inverse_hermite_transform MP mode ===
  - Test forward transform with use_mpmath=True
  - Reconstruct and verify at test points

=== TEST 9: Challenging integrand — oscillatory function ===
  - ∫ cos(x) · e^{-x²} dx = sqrt(π)·e^{-1/4} (known analytical result)
  - Verify with integrate_mp() at high precision

=== SUMMARY ===
  - Print PASS/FAIL summary using _pass_fail helper
```

---

## 4. Plan: `laguerre_tests/test_laguerre_05_integration_mp.py`

Create a comprehensive MP test file following the chebyshev pattern.

```
=== TEST 1: Gauss-Laguerre Quadrature (alpha=0) — ∫₀^∞ [1/(1+x²)] · e^{-x} dx ===
  - Use LaguerreQuadrature(n, alpha=0.0, use_mpmath=True)
  - Verify convergence for n in [8, 16, 32]
  - Compare against known reference value

=== TEST 2: integrate_mp() — full mpmath pipeline ===
  - Use .integrate_mp(lambda x: mp.mpf(1)/(1+x**2), dps=80)
  - Verify return type is mp.mpf (not float)
  - Assert result within tolerance

=== TEST 3: Gauss-Laguerre — ∫₀^∞ e^{-x} dx = 1 (f(x)=1, weight built-in) ===
  - f(x) = 1, integral of weight function over [0,∞) = 1
  - Verify for n in [8, 16, 32]

=== TEST 4: Generalized Gauss-Laguerre (alpha=0.5) — ∫₀^∞ x^{0.5} e^{-x} dx = Γ(1.5) ===
  - f(x) = 1 with alpha=0.5, expected result = Γ(alpha+1) = Γ(1.5) = sqrt(π)/2
  - Verify for n in [8, 16, 32]

=== TEST 5: Generalized Gauss-Laguerre (alpha=1.5) ===
  - ∫₀^∞ x^{1.5} e^{-x} dx = Γ(2.5) = 1.5·Γ(1.5) = 0.75·sqrt(π)
  - Verify with use_mpmath=True

=== TEST 6: Laguerre Projection (function_projection) ===
  - Project f(x) = exp(-2x) using function_projection(f, max_n=15, alpha=0.0)
  - Reconstruct and verify at test points in [0, ∞)

=== TEST 7: Precision convergence (dps=50 vs dps=200) ===
  - Integrate a challenging function using integrate_mp() with varying dps
  - Compare against mpmath quad reference
  - Assert that higher dps yields more correct digits

=== TEST 8: Nodes and weights properties ===
  - Verify nodes are sorted ascending
  - Verify all nodes positive (domain is [0, ∞))
  - Verify all weights positive
  - Verify sum of weights ≈ Γ(alpha+1) for f(x)=1

=== TEST 9: Challenging integrand — oscillatory function ===
  - ∫₀^∞ cos(x) · e^{-x} dx = 0.5 (known Laplace transform result)
  - Verify with integrate_mp() at high precision

=== SUMMARY ===
  - Print PASS/FAIL summary using _pass_fail helper
```

---

## 5. Plan: `legendre_tests/test_legendre_07_integration_mp.py`

Create a comprehensive MP test file following the chebyshev pattern, building on existing test_legendre_06 but adding structured PASS/FAIL assertions and broader coverage.

Note: Legendre already has partial MP coverage in test_legendre_05 (precision comparison) and test_legendre_06 (HighPrecisionGaussLegendre). This new file will be the definitive MP test suite with explicit assertions.

```
=== TEST 1: HighPrecisionGaussLegendre — ∫_{-1}^{1} e^x dx = e - 1/e ===
  - Use HighPrecisionGaussLegendre(n, dps=80)
  - Verify convergence for n in [8, 16, 32]
  - Assert: n=32 result within tolerance of exact value

=== TEST 2: LegendreQuadrature.integrate_mp() — full mpmath pipeline ===
  - Use LegendreQuadrature(n, use_mpmath=True, dps=80).integrate_mp(f)
  - Verify return type is mp.mpf (not float)
  - Assert result within tolerance

=== TEST 3: Gauss-Legendre — polynomial integrands on [-1, 1] ===
  - ∫ x^k dx for k in [0, 1, 2, 3, 4] (known values: 2/(k+1) for even, 0 for odd)
  - Verify exactness for low-degree polynomials

=== TEST 4: Arbitrary interval [a, b] via integrate_mp() ===
  - ∫₀¹ e^x dx = e - 1 using integrate_mp(f, a=0, b=1)
  - ∫₀^{π} sin(x) dx = 2 using integrate_mp(f, a=0, b=pi)

=== TEST 5: Precision convergence (dps=30 vs dps=120) ===
  - Integrate exp(x) with HighPrecisionGaussLegendre at varying dps
  - Compare against mpmath quad reference at dps=150
  - Count correct significant digits, assert monotonic improvement

=== TEST 6: Node count convergence (n=8 to n=80 at fixed dps=80) ===
  - Verify accuracy improves with more nodes up to dps limit

=== TEST 7: Nodes and weights properties ===
  - Verify nodes sorted ascending in (-1, 1)
  - Verify all weights positive
  - Verify sum of weights = 2.0 (integral of weight=1 over [-1,1])

=== TEST 8: gauss_legendre_high_precision factory function ===
  - Test factory returns HighPrecisionGaussLegendre instance
  - Verify integration result matches direct construction

=== TEST 9: Challenging integrand — oscillatory function ===
  - ∫_{-1}^{1} sin(20x)·e^x dx (known analytical formula)
  - Verify with HighPrecisionGaussLegendre at high precision

=== TEST 10: Sympy expression integration via lambdify ===
  - Pass a sympy expression to HighPrecisionGaussLegendre.integrate()
  - Verify automatic lambdification works correctly

=== SUMMARY ===
  - Print PASS/FAIL summary using _pass_fail helper
```

---

## 6. Common Test Pattern (from chebyshev reference)

Each MP test file should follow this structure:

```python
"""Test suite for <family>/integration.py — High-precision <Family> quadrature."""
from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

sys.path.insert(0, "..")

from <family>.integration import (
    # Classes to test
    ...
)

# ── Helpers ───────────────────────────────────────────────────────────

def _mpf(f):
    """Wrap a float-accepting callable so it works with mp.mpf inputs."""
    return lambda x: mp.mpf(f(float(x)))

def _pass_fail(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition

_all_pass = True

# ... test sections with _all_pass &= _pass_fail(...) ...

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if _all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — review output above")
print("=" * 60)
```

---

## 7. Implementation Order

1. **hermite_tests/test_hermite_05_integration_mp.py** — No MP tests exist at all (highest priority gap)
2. **laguerre_tests/test_laguerre_05_integration_mp.py** — No MP tests exist at all (highest priority gap)  
3. **legendre_tests/test_legendre_07_integration_mp.py** — Partial coverage exists, needs structured PASS/FAIL suite

---

## 8. Key Differences from Chebyshev Pattern

| Aspect | Chebyshev | Hermite/Laguerre/Legendre |
|--------|-----------|--------------------------|
| MP module | Separate `integration_mp.py` | Built into `integration.py` |
| Import source | `from chebyshev.integration_mp import ...` | `from hermite.integration import ...` etc. |
| Node/weight access | Public `_nodes`, `_weights` attributes | Varies: some have `.nodes` property, others `_nodes` |
| Interval support | Clenshaw-Curtis on [a,b] via `clencurt_mp_interval` | Hermite/Laguerre: infinite domain (no interval mapping). Legendre: arbitrary [a,b] via `integrate_mp(a, b)` |
| Weight function | 1/sqrt(1-x²) or sqrt(1-x²) | e^{-x²} (Hermite), x^α·e^{-x} (Laguerre), 1 (Legendre) |

---

## 9. Analytical Reference Values for Tests

### Hermite (weight: e^{-x²}, domain: (-∞, ∞))
- ∫ e^{-x²} dx = sqrt(π)
- ∫ x²·e^{-x²} dx = sqrt(π)/2
- ∫ x^{2k+1}·e^{-x²} dx = 0 (odd symmetry)
- ∫ e^{-ax²}·e^{-x²} dx = sqrt(π/(a+1)) for a > -1

### Laguerre (weight: x^α·e^{-x}, domain: [0, ∞))
- ∫ x^α·e^{-x} dx = Γ(α+1)
- ∫ e^{-ax}·e^{-x} dx = 1/(a+1) for a > -1 (alpha=0)
- ∫ cos(x)·e^{-x} dx = 1/2 (Laplace transform, alpha=0)

### Legendre (weight: 1, domain: [-1, 1])
- ∫ e^x dx = e - 1/e
- ∫ x^k dx = 2/(k+1) for even k, 0 for odd k
- ∫ sin(20x)·e^x dx = [2cosh(1)·sin(20) - 40·sinh(1)·cos(20)] / 401