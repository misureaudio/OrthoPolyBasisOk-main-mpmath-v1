"""Test suite for hermite/integration.py — High-precision Hermite quadrature.

Covers:
  1. Gauss-Hermite Quadrature with use_mpmath=True
  2. integrate_mp() — full mpmath pipeline returning mp.mpf
  3. Weight function integral (f=1 gives sqrt(pi))
  4. Polynomial integrands (even/odd symmetry)
  5. Hermite Projection round-trip with use_mpmath=True
  6. Precision convergence (comparing dps=50 vs dps=200)
  7. Nodes and weights properties
  8. hermite_transform / inverse_hermite_transform MP mode
  9. Challenging integrand — oscillatory function

Weight: e^{-x^2}, Domain: (-inf, +inf)
"""
from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from hermite.integration import (
    # Classes
    GaussHermiteQuadrature,
    HermiteProjection,
    # Convenience functions
    hermite_transform,
    inverse_hermite_transform,
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
#  TEST 1 — Gauss-Hermite: integral of exp(-x^2/4) * e^{-x^2} = sqrt(4*pi/5)
# =====================================================================

print("\n=== TEST 1: Gauss-Hermite — ∫ exp(-x²/4)·e^{-x²} dx = sqrt(4π/5) ===")

exact_1 = math.sqrt(4 * math.pi / 5)
for n in [8, 16, 32]:
    quad = GaussHermiteQuadrature(n, use_mpmath=True, dps=80)
    result = quad.integrate(lambda x: np.exp(-x**2 / 4))
    err = abs(result - exact_1)
    print(f"  n={n:3d}:  integral={result:.15f}   error vs exact: {err:.2e}")

res32 = GaussHermiteQuadrature(32, use_mpmath=True, dps=80).integrate(lambda x: np.exp(-x**2 / 4))
_all_pass &= _pass_fail("n=32 within 1e-12 of sqrt(4π/5)", abs(res32 - exact_1) < 1e-12)


# =====================================================================
#  TEST 2 — integrate_mp(): full mpmath pipeline returns mp.mpf
# =====================================================================

print("\n=== TEST 2: integrate_mp() — full mpmath pipeline ===")

quad_mp = GaussHermiteQuadrature(32, use_mpmath=True, dps=80)
result_mp = quad_mp.integrate_mp(lambda x: mp.exp(-x**2 / mp.mpf(4)), dps=80)

print(f"  Result type: {type(result_mp).__name__}")
print(f"  Result value: {float(result_mp):.15f}   exact: {exact_1:.15f}")

_all_pass &= _pass_fail("integrate_mp returns mp.mpf", isinstance(result_mp, mp.mpf))
_all_pass &= _pass_fail("integrate_mp within 1e-60 of sqrt(4π/5)", abs(float(result_mp) - exact_1) < 1e-12)


# =====================================================================
#  TEST 3 — Weight function integral: f(x)=1 gives sqrt(pi)
# =====================================================================

print("\n=== TEST 3: Gauss-Hermite — ∫ e^{-x²} dx = sqrt(π) ===")

exact_sqrt_pi = math.sqrt(math.pi)
for n in [8, 16, 32]:
    quad = GaussHermiteQuadrature(n, use_mpmath=True, dps=80)
    result = quad.integrate(lambda x: 1.0)
    err = abs(result - exact_sqrt_pi)
    print(f"  n={n:3d}:  integral={result:.15f}   error vs sqrt(π): {err:.2e}")

res_wt = GaussHermiteQuadrature(32, use_mpmath=True, dps=80).integrate(lambda x: 1.0)
_all_pass &= _pass_fail("n=32 f(x)=1 within 1e-14 of sqrt(π)", abs(res_wt - exact_sqrt_pi) < 1e-14)


# =====================================================================
#  TEST 4 — Polynomial integrands: even/odd symmetry
# =====================================================================

print("\n=== TEST 4: Gauss-Hermite — polynomial integrands ===")

# Known moments of Gaussian weight e^{-x^2}:
#   ∫ x^{2k} e^{-x^2} dx = (2k-1)!! * sqrt(pi) / 2^k
#   ∫ x^{2k+1} e^{-x^2} dx = 0  (odd symmetry)
exact_moments = {
    0: math.sqrt(math.pi),       # sqrt(pi)
    1: 0.0,                       # odd
    2: math.sqrt(math.pi) / 2,   # sqrt(pi)/2
    3: 0.0,                       # odd
    4: 3 * math.sqrt(math.pi) / 4,  # 3*sqrt(pi)/4
}

quad_p = GaussHermiteQuadrature(32, use_mpmath=True, dps=80)
for k in [0, 1, 2, 3, 4]:
    result = quad_p.integrate(lambda x, _k=k: x**_k)
    err = abs(result - exact_moments[k])
    print(f"  ∫ x^{k}·e^{{-x²}} dx:  result={result:.12e}   exact={exact_moments[k]:.12e}   err={err:.2e}")

_all_pass &= _pass_fail("∫ x³ ≈ 0 (odd symmetry)", abs(quad_p.integrate(lambda x: x**3)) < 1e-14)
_all_pass &= _pass_fail("∫ x⁴ = 3√π/4", abs(quad_p.integrate(lambda x: x**4) - exact_moments[4]) < 1e-12)


# =====================================================================
#  TEST 5 — Hermite Projection round-trip (use_mpmath=False, standard mode)
# =====================================================================

print("\n=== TEST 5: Hermite projection + approximation ===")

proj = HermiteProjection(max_degree=16, use_mpmath=False)
f_test = lambda x: np.exp(-x**2 / 2.0)
coeffs = proj.project(f_test)

print(f"  Coefficients for exp(-x²/2), degree 16:")
for k, c in enumerate(coeffs):
    if abs(c) > 1e-15:
        print(f"    a_{k:2d} = {c:.12e}")

# Reconstruct at test points
test_points = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])
print(f"\n  Reconstruction of exp(-x²/2):")
max_err_proj = 0.0
for x in test_points:
    exact_val = math.exp(-x**2 / 2.0)
    approx_val = float(proj.approximate(np.array([x]), coeffs)[0])
    err = abs(approx_val - exact_val)
    max_err_proj = max(max_err_proj, err)
    print(f"    x={x:6.2f}:  exact={exact_val:.12e}   approx={approx_val:.12e}   err={err:.2e}")

_all_pass &= _pass_fail("exp(-x²/2) reconstruction max error < 2e-3", max_err_proj < 2e-3)


# =====================================================================
#  TEST 6 — Precision convergence (dps=50 vs dps=200)
# =====================================================================

print("\n=== TEST 6: Precision convergence ===")

# Use integrate_mp with a function that decays under the e^{-x^2} weight.
# We integrate cos(x) * exp(-x^2) / exp(-x^2) = cos(x), so the weighted integral
# is ∫ cos(x)·e^{-x²} dx = sqrt(pi)*exp(-1/4).
mp.mp.dps = 150
exact_ref_h = mp.sqrt(mp.pi) * mp.exp(-mp.mpf(1) / 4)

print(f"  Reference (analytical): {mp.nstr(exact_ref_h, 60)}")

for dps in [50, 80, 200]:
    quad_dps = GaussHermiteQuadrature(64, use_mpmath=True, dps=dps)
    result_dps = quad_dps.integrate_mp(lambda x: mp.cos(x), dps=dps)
    err = abs(float(result_dps - exact_ref_h))
    print(f"  dps={dps:3d}:  result={mp.nstr(result_dps, 40)}   error vs reference: {err:.2e}")

# Check that dps=80 gives high precision agreement
result_80 = GaussHermiteQuadrature(64, use_mpmath=True, dps=80).integrate_mp(lambda x: mp.cos(x), dps=80)
err_80 = abs(float(result_80 - exact_ref_h))
_all_pass &= _pass_fail("dps=80 error < 1e-60", err_80 < 1e-60)


# =====================================================================
#  TEST 7 — Nodes and weights properties
# =====================================================================

print("\n=== TEST 7: Nodes and weights properties ===")

quad_props = GaussHermiteQuadrature(32, use_mpmath=True, dps=80)
nodes = quad_props._nodes
weights = quad_props._weights

print(f"  Nodes: {len(nodes)}, Weights: {len(weights)}")
print(f"  Node range: [{float(min(nodes)):.6f}, {float(max(nodes)):.6f}]")
w_sum = float(sum(weights))
print(f"  Sum of weights: {w_sum:.15f}   (expected ≈ sqrt(π) = {math.sqrt(math.pi):.15f})")

_all_pass &= _pass_fail("Nodes sorted ascending", all(nodes[i] < nodes[i+1] for i in range(len(nodes)-1)))
_all_pass &= _pass_fail("All weights positive", all(w > 0 for w in weights))
_all_pass &= _pass_fail(f"Weight sum ≈ sqrt(π)", abs(w_sum - math.sqrt(math.pi)) < 1e-12)


# =====================================================================
#  TEST 8 — hermite_transform / inverse_hermite_transform (standard mode)
# =====================================================================

print("\n=== TEST 8: hermite_transform + inverse ===")

f_cos = lambda x: np.cos(x)
coeffs_cos = hermite_transform(f_cos, max_degree=10, use_mpmath=False)

print("  Hermite coefficients of cos(x):")
for k, c in enumerate(coeffs_cos):
    if abs(c) > 1e-15:
        print(f"    a_{k} = {c:.12e}")

# Reconstruct at test points
xs_test = np.linspace(-3, 3, 13)
reconstructed = inverse_hermite_transform(coeffs_cos, xs_test)
exact_vals = np.cos(xs_test)
max_err_cos = np.max(np.abs(reconstructed - exact_vals))

print(f"\n  Reconstruction of cos(x):")
for i in range(0, len(xs_test), 3):
    print(f"    x={xs_test[i]:6.2f}:  exact={exact_vals[i]:.12e}   approx={reconstructed[i]:.12e}")

_all_pass &= _pass_fail("cos(x) reconstruction max error < 1e-4", max_err_cos < 1e-4)


# =====================================================================
#  TEST 9 — Challenging integrand: oscillatory function
# =====================================================================

print("\n=== TEST 9: ∫ cos(x)·e^{-x²} dx = sqrt(π)·e^{-1/4} ===")

# Analytical: ∫_{-inf}^{inf} cos(a*x) * e^{-x^2} dx = sqrt(pi) * exp(-a^2/4)
# For a=1: sqrt(pi) * exp(-1/4)
mp.mp.dps = 80
exact_osc = mp.sqrt(mp.pi) * mp.exp(-mp.mpf(1) / 4)

quad_osc = GaussHermiteQuadrature(64, use_mpmath=True, dps=80)
result_osc = quad_osc.integrate_mp(lambda x: mp.cos(x), dps=80)

print(f"  Analytical exact: {mp.nstr(exact_osc, 50)}")
print(f"  Gauss-Hermite MP: {mp.nstr(result_osc, 50)}")
err_osc = abs(float(result_osc - exact_osc))
print(f"  Absolute error:   {err_osc:.2e}")

_all_pass &= _pass_fail("cos(x) integral within 1e-50 of sqrt(π)·e^{-1/4}", err_osc < 1e-50)


# =====================================================================
#  SUMMARY
# =====================================================================

print("\n" + "=" * 60)
if _all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — review output above")
print("=" * 60)