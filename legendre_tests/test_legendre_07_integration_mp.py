"""Test suite for legendre/integration.py — High-precision Legendre quadrature.

Covers:
  1. HighPrecisionGaussLegendre: true arbitrary-precision integration
  2. LegendreQuadrature.integrate_mp(): full mpmath pipeline returning mp.mpf
  3. Polynomial integrands on [-1, 1] (exactness for low-degree polynomials)
  4. Arbitrary interval [a, b] via integrate_mp()
  5. Precision convergence (dps=30 vs dps=120)
  6. Node count convergence (n=8 to n=80 at fixed dps=80)
  7. Nodes and weights properties
  8. gauss_legendre_high_precision factory function
  9. Challenging integrand — oscillatory function
 10. Sympy expression integration via lambdify

Weight: 1, Domain: [-1, +1]
"""
from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from legendre.integration import (
    # Classes
    HighPrecisionGaussLegendre,
    LegendreQuadrature,
    # Convenience functions
    gauss_legendre_high_precision,
)

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

# ── Helpers ───────────────────────────────────────────────────────────

def _mpf(f):
    """Wrap a float-accepting callable so it works with mp.mpf inputs."""
    return lambda x: mp.mpf(f(float(x)))


def _pass_fail(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


_all_pass = True

# =====================================================================
#  TEST 1 — HighPrecisionGaussLegendre: ∫_{-1}^{1} e^x dx = e - 1/e
# =====================================================================

print("\n=== TEST 1: HighPrecisionGaussLegendre — ∫ eˣ dx = e - 1/e ===")

mp.mp.dps = 80
exact_1 = mp.exp(1) - mp.exp(-1)

for n in [8, 16, 32]:
    q = HighPrecisionGaussLegendre(n, dps=80)
    result = q.integrate(lambda x: mp.exp(x))
    err = abs(float(result - exact_1))
    print(f"  n={n:3d}:  integral={float(result):.15f}   error vs exact: {err:.2e}")

res32 = HighPrecisionGaussLegendre(32, dps=80).integrate(lambda x: mp.exp(x))
_all_pass &= _pass_fail("n=32 within 1e-60 of e - 1/e", abs(float(res32 - exact_1)) < 1e-60)


# =====================================================================
#  TEST 2 — LegendreQuadrature.integrate_mp(): full mpmath pipeline
# =====================================================================

print("\n=== TEST 2: integrate_mp() — full mpmath pipeline ===")

q_mp = LegendreQuadrature(32, use_mpmath=True, dps=80)
result_mp = q_mp.integrate_mp(lambda x: mp.exp(x))

print(f"  Result type: {type(result_mp).__name__}")
print(f"  Result value: {float(result_mp):.15f}   exact: {float(exact_1):.15f}")

_all_pass &= _pass_fail("integrate_mp returns mp.mpf", isinstance(result_mp, mp.mpf))
_all_pass &= _pass_fail("integrate_mp within 1e-60 of e - 1/e", abs(float(result_mp - exact_1)) < 1e-60)


# =====================================================================
#  TEST 3 — Polynomial integrands: ∫ x^k dx on [-1, 1]
# =====================================================================

print("\n=== TEST 3: Gauss-Legendre — polynomial integrands ===")

# Known values: ∫_{-1}^{1} x^k dx = 2/(k+1) for even k, 0 for odd k
exact_moments = {
    0: mp.mpf(2),          # ∫ 1 dx = 2
    1: mp.mpf(0),           # odd
    2: mp.mpf(2) / 3,      # ∫ x² dx = 2/3
    3: mp.mpf(0),           # odd
    4: mp.mpf(2) / 5,      # ∫ x⁴ dx = 2/5
}

q_p = HighPrecisionGaussLegendre(16, dps=80)
for k in [0, 1, 2, 3, 4]:
    result = q_p.integrate(lambda x, _k=k: x**_k)
    err = abs(float(result - exact_moments[k]))
    print(f"  ∫ x^{k} dx:  result={float(result):.15f}   exact={float(exact_moments[k]):.15f}   err={err:.2e}")

_all_pass &= _pass_fail("∫ x³ = 0 (odd symmetry)", abs(float(q_p.integrate(lambda x: x**3))) < 1e-70)
_all_pass &= _pass_fail("∫ x⁴ = 2/5", abs(float(q_p.integrate(lambda x: x**4)) - float(exact_moments[4])) < 1e-70)


# =====================================================================
#  TEST 4 — Arbitrary interval [a, b] via integrate_mp()
# =====================================================================

print("\n=== TEST 4: Arbitrary interval [a, b] ===")

# ∫₀¹ e^x dx = e - 1
mp.mp.dps = 80
exact_4a = mp.exp(1) - 1

q_ab = LegendreQuadrature(32, use_mpmath=True, dps=80)
result_4a = q_ab.integrate_mp(lambda x: mp.exp(x), a=0, b=1)

print(f"  ∫₀¹ eˣ dx:")
print(f"    Result: {float(result_4a):.15f}   Exact: {float(exact_4a):.15f}")
_all_pass &= _pass_fail("∫₀¹ eˣ within 1e-60 of e - 1", abs(float(result_4a - exact_4a)) < 1e-60)

# ∫₀^π sin(x) dx = 2
exact_4b = mp.mpf(2)
result_4b = q_ab.integrate_mp(lambda x: mp.sin(x), a=0, b=float(mp.pi))

print(f"  ∫₀^π sin(x) dx:")
print(f"    Result: {float(result_4b):.15f}   Exact: 2.0")
# Note: float(mp.pi) loses precision for the interval boundary, so tolerance is relaxed
_all_pass &= _pass_fail("∫₀^π sin(x) within 1e-14 of 2", abs(float(result_4b - exact_4b)) < 1e-14)


# =====================================================================
#  TEST 5 — Precision convergence (dps=30 vs dps=120)
# =====================================================================

print("\n=== TEST 5: Precision convergence ===")

mp.mp.dps = 150
exact_ref = mp.exp(1) - mp.exp(-1)

print(f"  Reference (mpmath dps=150): {mp.nstr(exact_ref, 60)}")

for dps in [30, 50, 80, 120]:
    q_dps = HighPrecisionGaussLegendre(50, dps=dps)
    result_dps = q_dps.integrate(lambda x: mp.exp(x))
    err = abs(float(result_dps - exact_ref))
    print(f"  dps={dps:3d}:  error vs reference: {err:.2e}")

# Check that dps=80 gives at least ~75 correct digits
result_80 = HighPrecisionGaussLegendre(50, dps=80).integrate(lambda x: mp.exp(x))
err_80 = abs(float(result_80 - exact_ref))
_all_pass &= _pass_fail("dps=80 error < 1e-70", err_80 < 1e-70)


# =====================================================================
#  TEST 6 — Node count convergence (n=8 to n=80 at fixed dps=80)
# =====================================================================

print("\n=== TEST 6: Node count convergence ===")

mp.mp.dps = 80
exact_n = mp.exp(1) - mp.exp(-1)

for n in [8, 16, 32, 50, 80]:
    q_n = HighPrecisionGaussLegendre(n, dps=80)
    result_n = q_n.integrate(lambda x: mp.exp(x))
    err = abs(float(result_n - exact_n))
    print(f"  n={n:3d}:  error vs reference: {err:.2e}")

_all_pass &= _pass_fail("n=80 error < 1e-70", abs(float(HighPrecisionGaussLegendre(80, dps=80).integrate(lambda x: mp.exp(x)) - exact_n)) < 1e-70)


# =====================================================================
#  TEST 7 — Nodes and weights properties
# =====================================================================

print("\n=== TEST 7: Nodes and weights properties ===")

q_props = HighPrecisionGaussLegendre(32, dps=80)
nodes = q_props.nodes
weights = q_props.weights

print(f"  Nodes: {len(nodes)}, Weights: {len(weights)}")
print(f"  Node range: [{float(min(nodes)):.6f}, {float(max(nodes)):.6f}]")
w_sum = float(sum(weights))
print(f"  Sum of weights: {w_sum:.15f}   (expected ≈ 2.0)")

_all_pass &= _pass_fail("Nodes sorted ascending", all(float(nodes[i]) < float(nodes[i+1]) for i in range(len(nodes)-1)))
_all_pass &= _pass_fail("All nodes in (-1, 1)", all(-1 < float(x) < 1 for x in nodes))
_all_pass &= _pass_fail("All weights positive", all(float(w) > 0 for w in weights))
_all_pass &= _pass_fail(f"Weight sum ≈ 2.0", abs(w_sum - 2.0) < 1e-70)


# =====================================================================
#  TEST 8 — gauss_legendre_high_precision factory function
# =====================================================================

print("\n=== TEST 8: Factory function ===")

q_factory = gauss_legendre_high_precision(32, dps=60)
result_f = q_factory.integrate(lambda x: mp.exp(x))

mp.mp.dps = 60
exact_f = mp.exp(1) - mp.exp(-1)
err_f = abs(float(result_f - exact_f))

print(f"  Type: {type(q_factory).__name__}")
print(f"  Result: {float(result_f):.15f}   Exact: {float(exact_f):.15f}   err={err_f:.2e}")

_all_pass &= _pass_fail("Factory returns HighPrecisionGaussLegendre", isinstance(q_factory, HighPrecisionGaussLegendre))
_all_pass &= _pass_fail("Factory result within 1e-50 of e - 1/e", err_f < 1e-50)


# =====================================================================
#  TEST 9 — Challenging integrand: oscillatory function
# =====================================================================

print("\n=== TEST 9: ∫ sin(20x)·eˣ dx (oscillatory) ===")

mp.mp.dps = 80
a = mp.mpf(20)
exact_osc = (2*mp.cosh(1)*mp.sin(a) - 2*a*mp.sinh(1)*mp.cos(a)) / (1 + a**2)

q_chal = HighPrecisionGaussLegendre(80, dps=60)
result_chal = q_chal.integrate(lambda x: mp.sin(20 * x) * mp.exp(x))

print(f"  Analytical exact: {mp.nstr(exact_osc, 50)}")
print(f"  Gauss-Legendre MP: {mp.nstr(result_chal, 50)}")
err_osc = abs(float(result_chal - exact_osc))
print(f"  Absolute error:   {err_osc:.2e}")

_all_pass &= _pass_fail("sin(20x)·eˣ within 1e-40 of analytical", err_osc < 1e-40)


# =====================================================================
#  TEST 10 — Sympy expression integration via lambdify
# =====================================================================

print("\n=== TEST 10: Sympy expression integration ===")

if HAS_SYMPY:
    x_sym = sp.Symbol('x')
    expr = sp.exp(x_sym)

    q_sym = HighPrecisionGaussLegendre(32, dps=80)
    result_sym = q_sym.integrate(expr)

    mp.mp.dps = 80
    exact_sym = mp.exp(1) - mp.exp(-1)
    err_sym = abs(float(result_sym - exact_sym))

    print(f"  Expression: {expr}")
    print(f"  Result: {float(result_sym):.15f}   Exact: {float(exact_sym):.15f}   err={err_sym:.2e}")

    _all_pass &= _pass_fail("Sympy exp(x) integration within 1e-60", err_sym < 1e-60)
else:
    print("  [SKIP] Sympy not installed")


# =====================================================================
#  SUMMARY
# =====================================================================

print("\n" + "=" * 60)
if _all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — review output above")
print("=" * 60)