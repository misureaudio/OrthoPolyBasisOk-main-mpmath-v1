"""Test suite for chebyshev/integration_mp.py — High-precision Chebyshev quadrature.

Covers:
  1. Gauss-Chebyshev Quadrature (Type I and Type II)
  2. Clenshaw-Curtis MP quadrature on [-1, 1]
  3. Clenshaw-Curtis MP quadrature on arbitrary intervals [a, b]
  4. Chebyshev projection / approximation round-trip
  5. Precision convergence (comparing dps=50 vs dps=200)
  6. Float64 bridge functions
"""
from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from chebyshev.integration_mp import (
    # Classes
    GaussChebyshevQuadrature,
    ClenshawCurtisMP,
    ChebyshevProjectionMP,
    # Convenience functions
    clencurt_mp,
    clencurt_mp_interval,
    gauss_chebyshev_mp,
    chebyshev_transform_mp,
    inverse_chebyshev_transform_mp,
    get_nodes_weights_mp,
    map_nodes_to_interval_mp,
    clencurt_quadrature_float,
    chebyshev_transform_float,
)

# ── Helpers ───────────────────────────────────────────────────────────

def _mpf(f):
    """Wrap a float-accepting callable so it works with mp.mpf inputs."""
    return lambda x: mp.mpf(f(float(x)))


def _pass_fail(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# =====================================================================
#  TEST 1 — Gauss-Chebyshev Type I: integral of 1 / sqrt(1-x^2) = pi
# =====================================================================

print("\n=== TEST 1: Gauss-Chebyshev Type I — ∫ dx/sqrt(1-x²) = π ===")

f_typeI = lambda x: mp.mpf(1)  # weight is built-in, so f=1 gives the integral of w(x)
for n in [8, 16, 32]:
    quad = GaussChebyshevQuadrature(n, quad_type="I", dps=80)
    result = quad.integrate(f_typeI)
    err = float(abs(result - mp.pi))
    print(f"  n={n:3d}:  integral={float(result):.15f}   error vs π: {err:.2e}")

_all_pass = True
quad32 = GaussChebyshevQuadrature(32, quad_type="I", dps=80)
res32 = float(quad32.integrate(f_typeI))
_all_pass &= _pass_fail("n=32 result within 1e-14 of π", abs(res32 - math.pi) < 1e-14)

# ── Nodes are in (-1, 1) and sorted ascending ────────────────────────
#nodes = quad32.nodes
nodes = quad32._nodes
_all_pass &= _pass_fail("Nodes sorted ascending", all(nodes[i] < nodes[i+1] for i in range(len(nodes)-1)))
_all_pass &= _pass_fail("All nodes in (-1, 1)", all(-1 < x < 1 for x in nodes))

# ── All weights equal π/n for Type I (identical regardless of sort) ──
with mp.workdps(80):
    expected_w = mp.pi / 32
# weights_ok = all(abs(w - expected_w) < mp.mpf("1e-70") for w in quad32.weights)
weights_ok = all(abs(w - expected_w) < mp.mpf("1e-70") for w in quad32._weights)
_all_pass &= _pass_fail(f"All weights ≈ π/32", weights_ok)


# =====================================================================
#  TEST 2 — Gauss-Chebyshev Type II: integral of sqrt(1-x^2) = pi/2
# =====================================================================

print("\n=== TEST 2: Gauss-Chebyshev Type II — ∫ sqrt(1-x²) dx = π/2 ===")

f_typeII = lambda x: mp.mpf(1)  # weight is built-in, so f=1 gives integral of w(x)=sqrt(1-x²)
for n in [8, 16, 32]:
    quad = GaussChebyshevQuadrature(n, quad_type="II", dps=80)
    result = quad.integrate(f_typeII)
    err = float(abs(result - mp.pi / 2))
    print(f"  n={n:3d}:  integral={float(result):.15f}   error vs π/2: {err:.2e}")

quad32_II = GaussChebyshevQuadrature(32, quad_type="II", dps=80)
res32_II = float(quad32_II.integrate(f_typeII))
_all_pass &= _pass_fail("n=32 result within 1e-14 of π/2", abs(res32_II - math.pi / 2) < 1e-14)


# =====================================================================
#  TEST 3 — Clenshaw-Curtis MP: ∫_{-1}^{1} x^k dx (known values)
# =====================================================================

print("\n=== TEST 3: Clenshaw-Curtis MP — polynomial integrals on [-1, 1] ===")

for k in [0, 2, 4, 6]:
    exact = mp.mpf(2) / (k + 1) if k % 2 == 0 else mp.mpf(0)
    f = _mpf(lambda x, _k=k: x ** _k)
    for n in [16, 32]:
        result = clencurt_mp(f, n, dps=80)
        err = float(abs(result - exact))
        print(f"  ∫ x^{k}, n={n}:  result={float(result):.15f}   exact={float(exact):.15f}   err={err:.2e}")

# Verify odd power integrates to ~0
res_odd = float(clencurt_mp(_mpf(lambda x: x ** 3), 32, dps=80))
_all_pass &= _pass_fail("∫ x³ ≈ 0", abs(res_odd) < 1e-14)

# Verify ∫ 1 dx = 2
res_const = float(clencurt_mp(_mpf(lambda x: 1.0), 32, dps=80))
_all_pass &= _pass_fail("∫ 1 dx = 2", abs(res_const - 2.0) < 1e-14)


# =====================================================================
#  TEST 4 — Clenshaw-Curtis MP on arbitrary interval [a, b]
# =====================================================================

print("\n=== TEST 4: Clenshaw-Curtis MP — ∫₀¹⁰ t² dt = 1000/3 ===")

f_t2 = _mpf(lambda t: t ** 2)
exact_10 = mp.mpf(1000) / 3

for n in [16, 32, 64]:
    result = clencurt_mp_interval(f_t2, 0, 10, n=n, dps=80)
    err = abs(float(result) - float(exact_10))
    print(f"  n={n:3d}:  result={float(result):.15f}   exact={float(exact_10):.15f}   err={err:.2e}")

res_t2 = float(clencurt_mp_interval(f_t2, 0, 10, n=64, dps=80))
_all_pass &= _pass_fail("∫₀¹⁰ t² within 1e-10 of 1000/3", abs(res_t2 - 1000/3) < 1e-10)


# ── ∫₀^π sin(t) dt = 2 ───────────────────────────────────────────────
print("\n=== TEST 4b: Clenshaw-Curtis MP — ∫₀^π sin(t) dt = 2 ===")

f_sin = _mpf(math.sin)
for n in [16, 32, 64]:
    result = clencurt_mp_interval(f_sin, 0, math.pi, n=n, dps=80)
    err = abs(float(result) - 2.0)
    print(f"  n={n:3d}:  result={float(result):.15f}   exact=2.0   err={err:.2e}")

res_sin = float(clencurt_mp_interval(f_sin, 0, math.pi, n=64, dps=80))
_all_pass &= _pass_fail("∫₀^π sin(t) within 1e-10 of 2", abs(res_sin - 2.0) < 1e-10)


# =====================================================================
#  TEST 5 — Chebyshev Projection / Approximation round-trip
# =====================================================================

print("\n=== TEST 5: Chebyshev projection + Clenshaw approximation ===")

# Project f(x) = exp(x), then reconstruct at several points
f_exp = _mpf(math.exp)
max_deg = 16

proj = ChebyshevProjectionMP(max_deg, dps=80)
coeffs = proj.project(f_exp)

print(f"  Coefficients (degree {max_deg}):")
for k, c in enumerate(coeffs):
    print(f"    a_{k:2d} = {float(c):.15e}")

# Reconstruct at test points
test_points = [-0.8, -0.3, 0.0, 0.5, 0.9]
print(f"\n  Reconstruction of exp(x):")
for x in test_points:
    exact_val = math.exp(x)
    approx_val = float(proj.approximate(x, coeffs))
    err = abs(approx_val - exact_val)
    print(f"    x={x:5.2f}:  exact={exact_val:.15f}   approx={approx_val:.15f}   err={err:.2e}")

# Check max error across test points
max_err = max(abs(float(proj.approximate(x, coeffs)) - math.exp(x)) for x in test_points)
_all_pass &= _pass_fail("exp(x) reconstruction max error < 1e-10", max_err < 1e-10)


# =====================================================================
#  TEST 6 — Precision convergence (dps=50 vs dps=200)
# =====================================================================

print("\n=== TEST 6: Precision convergence ===")

# Integrate exp(x)*sin(x) on [-1, 1] using mpmath quad as reference
f_complex = lambda x: mp.exp(x) * mp.sin(x)
reference = mp.quad(f_complex, [-1, 1])

print(f"  Reference (mpmath quad): {reference}")

for dps in [50, 80, 200]:
    with mp.workdps(dps + 20):
        ref_val = mp.quad(f_complex, [-1, 1])
    result = clencurt_mp(f_complex, n=64, dps=dps)
    err = abs(float(result - ref_val))
    print(f"  dps={dps:3d}:  result={float(result):.20f}   error vs reference: {err:.2e}")

# Check that dps=80 gives at least ~65 correct decimal digits
result_80 = clencurt_mp(f_complex, n=64, dps=80)
with mp.workdps(100):
    ref_100 = mp.quad(f_complex, [-1, 1])
err_80 = abs(float(result_80 - ref_100))
_all_pass &= _pass_fail("dps=80 error < 1e-65", err_80 < 1e-65)


# =====================================================================
#  TEST 7 — Convenience functions: chebyshev_transform_mp + inverse
# =====================================================================

print("\n=== TEST 7: chebyshev_transform_mp / inverse_chebyshev_transform_mp ===")

f_cos2 = lambda x: mp.cos(mp.mpf(2) * x)
# Use higher degree for better convergence of cos(2x) Chebyshev expansion
coeffs_7 = chebyshev_transform_mp(f_cos2, max_degree=20, dps=80)

print("  Coefficients of cos(2x):")
for k, c in enumerate(coeffs_7):
    if abs(float(c)) > 1e-30:
        print(f"    a_{k} = {float(c):.15f}")

# Reconstruct at x=0.5
approx_7 = float(inverse_chebyshev_transform_mp(coeffs_7, mp.mpf("0.5"), dps=80))
exact_7 = math.cos(2 * 0.5)
err_7 = abs(approx_7 - exact_7)
print(f"  cos(2*0.5): exact={exact_7:.15f}   approx={approx_7:.15f}   err={err_7:.2e}")
_all_pass &= _pass_fail("cos(2x) reconstruction at x=0.5 within 1e-8", err_7 < 1e-8)


# =====================================================================
#  TEST 8 — get_nodes_weights_mp and map_nodes_to_interval_mp
# =====================================================================

print("\n=== TEST 8: get_nodes_weights_mp + map_nodes_to_interval_mp ===")

nodes, weights = get_nodes_weights_mp(32, dps=80)
print(f"  Nodes: {len(nodes)}, Weights: {len(weights)}")
w_sum = sum(weights)
print(f"  Sum of weights: {float(w_sum):.15f}   (expected ≈ 2)")
_all_pass &= _pass_fail("Weight sum ≈ 2", abs(float(w_sum) - 2.0) < 1e-14)

# Map nodes to [3, 7]
# CC nodes are cos(k*pi/n), k=0..n → [1, ..., -1] (descending)
# So mapped[0] = (b-a)/2 * 1 + (b+a)/2 = b = 7
#    mapped[-1] = (b-a)/2 * (-1) + (b+a)/2 = a = 3
mapped = map_nodes_to_interval_mp(nodes, 3, 7, dps=80)
print(f"  Mapped node range: [{float(mapped[0]):.6f}, {float(mapped[-1]):.6f}]")
_all_pass &= _pass_fail("First mapped node ≈ 7 (from x=+1)", abs(float(mapped[0]) - 7.0) < 1e-14)
_all_pass &= _pass_fail("Last mapped node ≈ 3 (from x=-1)", abs(float(mapped[-1]) - 3.0) < 1e-14)


# =====================================================================
#  TEST 9 — Float64 bridge functions
# =====================================================================

print("\n=== TEST 9: Float64 bridge functions ===")

# clencurt_quadrature_float
res_bridge = clencurt_quadrature_float(lambda x: x ** 2, n=32, dps=80)
exact_bridge = 2.0 / 3.0
print(f"  ∫ x² on [-1,1]: result={res_bridge:.15f}   exact={exact_bridge:.15f}")
_all_pass &= _pass_fail("Float bridge: ∫ x² within 1e-14 of 2/3", abs(res_bridge - exact_bridge) < 1e-14)

# chebyshev_transform_float
coeffs_bridge = chebyshev_transform_float(math.exp, max_degree=8, dps=80)
print(f"  Chebyshev coeffs (exp, float64): {np.array2string(coeffs_bridge, precision=12)}")
_all_pass &= _pass_fail("Float bridge returns np.ndarray", isinstance(coeffs_bridge, np.ndarray))


# =====================================================================
#  TEST 10 — gauss_chebyshev_mp convenience function
# =====================================================================

print("\n=== TEST 10: gauss_chebyshev_mp convenience ===")

# ∫_{-1}^{1} 1/sqrt(1-x²) dx = π
res_gc = float(gauss_chebyshev_mp(lambda x: mp.mpf(1), n=32, quad_type="I", dps=80))
print(f"  Type I (f=1): {res_gc:.15f}   vs π={math.pi:.15f}")
_all_pass &= _pass_fail("gauss_chebyshev_mp Type I within 1e-14 of π", abs(res_gc - math.pi) < 1e-14)


# =====================================================================
#  SUMMARY
# =====================================================================

print("\n" + "=" * 60)
if _all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — review output above")
print("=" * 60)