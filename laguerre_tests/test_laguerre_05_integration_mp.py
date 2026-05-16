"""Test suite for laguerre/integration.py — High-precision Laguerre quadrature.

Covers:
  1. Gauss-Laguerre Quadrature (alpha=0) with use_mpmath=True
  2. integrate_mp() — full mpmath pipeline returning mp.mpf
  3. Weight function integral (f=1 gives Gamma(alpha+1))
  4. Generalized Gauss-Laguerre (alpha=0.5, alpha=1.5)
  5. Laguerre projection via function_projection
  6. Precision convergence (comparing dps=50 vs dps=200)
  7. Nodes and weights properties
  8. Challenging integrand — oscillatory function

Weight: x^alpha * e^{-x}, Domain: [0, +inf)
"""
from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from laguerre.integration import (
    # Classes
    LaguerreQuadrature,
    GeneralizedLaguerreBasis,
    # Convenience functions
    function_projection,
    function_approximation,
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

# =====================================================================
#  TEST 1 — Gauss-Laguerre (alpha=0): integral of 1/(1+x^2) * e^{-x}
# =====================================================================

print("\n=== TEST 1: Gauss-Laguerre (alpha=0) — ∫ [1/(1+x²)]·e⁻ˣ dx ===")

# Reference via mpmath quad at high precision
mp.mp.dps = 80
ref_1 = mp.quad(lambda x: mp.exp(-x) / (1 + x**2), [0, mp.inf])

for n in [8, 16, 32]:
    ql = LaguerreQuadrature(n, alpha=0.0, use_mpmath=True)
    result = ql.integrate(lambda x: 1.0 / (1.0 + x**2))
    err = abs(float(result) - float(ref_1))
    print(f"  n={n:3d}:  integral={float(result):.15f}   error vs ref: {err:.2e}")

res64 = LaguerreQuadrature(64, alpha=0.0, use_mpmath=True).integrate(lambda x: 1.0 / (1.0 + x**2))
_all_pass &= _pass_fail("n=64 within 1e-8 of reference", abs(float(res64) - float(ref_1)) < 1e-8)


# =====================================================================
#  TEST 2 — integrate_mp(): full mpmath pipeline returns mp.mpf
# =====================================================================

print("\n=== TEST 2: integrate_mp() — full mpmath pipeline ===")

ql_mp = LaguerreQuadrature(32, alpha=0.0)
result_mp = ql_mp.integrate_mp(lambda x: mp.mpf(1) / (1 + x**2), dps=80)

print(f"  Result type: {type(result_mp).__name__}")
print(f"  Result value: {float(result_mp):.15f}   ref: {float(ref_1):.15f}")

_all_pass &= _pass_fail("integrate_mp returns mp.mpf", isinstance(result_mp, mp.mpf))

# Use n=64 for tighter accuracy on this challenging integrand
ql_mp64 = LaguerreQuadrature(64, alpha=0.0)
result_mp64 = ql_mp64.integrate_mp(lambda x: mp.mpf(1) / (1 + x**2), dps=80)
_all_pass &= _pass_fail("integrate_mp within 1e-8 of reference", abs(float(result_mp64) - float(ref_1)) < 1e-8)


# =====================================================================
#  TEST 3 — Weight function integral: f(x)=1 gives Gamma(alpha+1) = 1 for alpha=0
# =====================================================================

print("\n=== TEST 3: Gauss-Laguerre (alpha=0) — ∫ e⁻ˣ dx = 1 ===")

for n in [8, 16, 32]:
    ql = LaguerreQuadrature(n, alpha=0.0, use_mpmath=True)
    result = ql.integrate(lambda x: 1.0)
    err = abs(result - 1.0)
    print(f"  n={n:3d}:  integral={result:.15f}   error vs 1: {err:.2e}")

res_wt = LaguerreQuadrature(32, alpha=0.0, use_mpmath=True).integrate(lambda x: 1.0)
_all_pass &= _pass_fail("n=32 f(x)=1 within 1e-10 of 1", abs(res_wt - 1.0) < 1e-10)


# =====================================================================
#  TEST 4 — Generalized (alpha=0.5): ∫ x^{0.5} e^{-x} dx = Gamma(1.5) = sqrt(pi)/2
# =====================================================================

print("\n=== TEST 4: Generalized Gauss-Laguerre (alpha=0.5) — ∫ x^0.5·e⁻ˣ dx ===")

exact_4 = math.sqrt(math.pi) / 2
for n in [8, 16, 32]:
    ql = LaguerreQuadrature(n, alpha=0.5, use_mpmath=True)
    result = ql.integrate(lambda x: 1.0)
    err = abs(result - exact_4)
    print(f"  n={n:3d}:  integral={result:.15f}   error vs Γ(1.5): {err:.2e}")

res_alpha_half = LaguerreQuadrature(32, alpha=0.5, use_mpmath=True).integrate(lambda x: 1.0)
_all_pass &= _pass_fail("alpha=0.5 f(x)=1 within 1e-10 of Γ(1.5)", abs(res_alpha_half - exact_4) < 1e-10)


# =====================================================================
#  TEST 5 — Generalized (alpha=1.5): ∫ x^{1.5} e^{-x} dx = Gamma(2.5) = 0.75*sqrt(pi)
# =====================================================================

print("\n=== TEST 5: Generalized Gauss-Laguerre (alpha=1.5) — ∫ x^1.5·e⁻ˣ dx ===")

exact_5 = 1.5 * math.sqrt(math.pi) / 2   # Gamma(2.5) = 3/4 * sqrt(pi)
for n in [8, 16, 32]:
    ql = LaguerreQuadrature(n, alpha=1.5, use_mpmath=True)
    result = ql.integrate(lambda x: 1.0)
    err = abs(result - exact_5)
    print(f"  n={n:3d}:  integral={result:.15f}   error vs Γ(2.5): {err:.2e}")

res_alpha_15 = LaguerreQuadrature(32, alpha=1.5, use_mpmath=True).integrate(lambda x: 1.0)
_all_pass &= _pass_fail("alpha=1.5 f(x)=1 within 1e-10 of Γ(2.5)", abs(res_alpha_15 - exact_5) < 1e-10)


# =====================================================================
#  TEST 6 — Laguerre projection via function_projection
# =====================================================================

print("\n=== TEST 6: Function projection (spectral expansion) ===")

f_proj = lambda x: np.exp(-2.0 * x)
coeffs = function_projection(f_proj, max_n=15, alpha=0.0)

print("  Coefficients for exp(-2x), standard Laguerre:")
for k, c in enumerate(coeffs):
    if abs(c) > 1e-15:
        print(f"    a_{k:2d} = {c:.12e}")

# Reconstruct using function_approximation
approx_fn = function_approximation(f_proj, max_n=15, alpha=0.0)
test_x = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
print(f"\n  Reconstruction of exp(-2x):")
max_err_proj = 0.0
for x in test_x:
    exact_val = math.exp(-2 * x)
    approx_val = float(approx_fn(x))
    err = abs(approx_val - exact_val)
    max_err_proj = max(max_err_proj, err)
    print(f"    x={x:5.1f}:  exact={exact_val:.12e}   approx={approx_val:.12e}   err={err:.2e}")

_all_pass &= _pass_fail("exp(-2x) reconstruction max error < 5e-3", max_err_proj < 5e-3)


# =====================================================================
#  TEST 7 — Precision convergence (dps=50 vs dps=200)
# =====================================================================

print("\n=== TEST 7: Precision convergence ===")

# Use integrate_mp with exp(-x/3): ∫₀^∞ e^{-x/3}·e^{-x} dx = 1/(1+1/3) = 3/4
mp.mp.dps = 150
exact_ref_l = mp.mpf(3) / 4

print(f"  Reference (analytical): {mp.nstr(exact_ref_l, 60)}")

for dps in [50, 80, 200]:
    ql_dps = LaguerreQuadrature(64, alpha=0.0)
    result_dps = ql_dps.integrate_mp(lambda x: mp.exp(-x / 3), dps=dps)
    err = abs(float(result_dps - exact_ref_l))
    print(f"  dps={dps:3d}:  result={mp.nstr(result_dps, 40)}   error vs reference: {err:.2e}")

# Check that dps=80 gives high precision agreement
result_80 = LaguerreQuadrature(64, alpha=0.0).integrate_mp(lambda x: mp.exp(-x / 3), dps=80)
err_80 = abs(float(result_80 - exact_ref_l))
_all_pass &= _pass_fail("dps=80 error < 1e-50", err_80 < 1e-50)


# =====================================================================
#  TEST 8 — Nodes and weights properties
# =====================================================================

print("\n=== TEST 8: Nodes and weights properties ===")

ql_props = LaguerreQuadrature(32, alpha=0.0, use_mpmath=True)
nodes = ql_props.nodes
weights = ql_props.weights

print(f"  Nodes: {len(nodes)}, Weights: {len(weights)}")
print(f"  Node range: [{float(min(nodes)):.6f}, {float(max(nodes)):.6f}]")
w_sum = float(np.sum(weights))
print(f"  Sum of weights: {w_sum:.15f}   (expected ≈ Γ(1) = 1.0)")

_all_pass &= _pass_fail("Nodes sorted ascending", all(nodes[i] < nodes[i+1] for i in range(len(nodes)-1)))
_all_pass &= _pass_fail("All nodes positive (domain [0,∞))", all(n > 0 for n in nodes))
_all_pass &= _pass_fail("All weights positive", all(w > 0 for w in weights))
_all_pass &= _pass_fail(f"Weight sum ≈ Γ(1) = 1.0", abs(w_sum - 1.0) < 1e-10)


# =====================================================================
#  TEST 9 — Challenging integrand: oscillatory function
# =====================================================================

print("\n=== TEST 9: ∫ cos(x)·e⁻ˣ dx = 1/2 (Laplace transform) ===")

mp.mp.dps = 80
exact_osc_l = mp.mpf(1) / 2   # ∫₀^∞ cos(x)·e^{-x} dx = Re[1/(1+i)] = 1/2

ql_osc = LaguerreQuadrature(64, alpha=0.0)
result_osc = ql_osc.integrate_mp(lambda x: mp.cos(x), dps=80)

print(f"  Analytical exact: {mp.nstr(exact_osc_l, 50)}")
print(f"  Gauss-Laguerre MP: {mp.nstr(result_osc, 50)}")
err_osc = abs(float(result_osc - exact_osc_l))
print(f"  Absolute error:   {err_osc:.2e}")

_all_pass &= _pass_fail("cos(x) integral within 1e-40 of 1/2", err_osc < 1e-40)


# =====================================================================
#  SUMMARY
# =====================================================================

print("\n" + "=" * 60)
if _all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — review output above")
print("=" * 60)