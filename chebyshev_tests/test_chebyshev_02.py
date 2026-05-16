"""Test Chebyshev Layer 2 — High Precision (mpmath).

Demonstrates arbitrary-precision evaluation, differentiation, integration,
and coefficient extraction for Chebyshev-T polynomials.
"""

from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

# import mpmath as mp
from chebyshev.high_precision import ChebyshevMPMath, get_mpmath_chebyshev_basis


def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


separator("LAYER 2 — High Precision Chebyshev-T (mpmath)")

# --- 1. High-precision evaluation near boundary ---
T30 = ChebyshevMPMath(30, dps=80)
x_val = "0.999"

val = T30.evaluate(x_val)
print(f"\nT_30({x_val})  [dps=80]:")
print(f"  {mp.nstr(val, 50)}")

# Verify against cos(30 * arccos(0.999))
mp.mp.dps = 80
expected = mp.cos(30 * mp.acos(mp.mpf(x_val)))
print(f"\n  Verification via cos(30·arccos({x_val})):")
print(f"  {mp.nstr(expected, 50)}")
tol_eval = mp.mpf('1e-75')
print(f"  Match (|diff| < 1e-75): {abs(val - expected) < tol_eval}")

# --- 2. High-precision derivative ---
separator("High-Precision Derivative  T_30'(x)")
deriv = T30.derivative(x_val)
print(f"\nT_30'({x_val}):")
print(f"  {mp.nstr(deriv, 50)}")

# Verify: T_n'(x) = n * U_{n-1}(x), where U is Chebyshev of second kind
expected_deriv = 30 * mp.chebyu(29, mp.mpf(x_val))
print(f"\n  Verification via 30·U_29({x_val}):")
print(f"  {mp.nstr(expected_deriv, 50)}")
tol_deriv = mp.mpf('1e-75')
print(f"  Match (|diff| < 1e-75): {abs(deriv - expected_deriv) < tol_deriv}")

# --- 3. High-precision integral ---
separator("High-Precision Integral  ∫₀^{0.5} T_30(t) dt")
integ = T30.integral(0.5)
print(f"\n∫₀⁰·⁵ T_30(t) dt:")
print(f"  {mp.nstr(integ, 50)}")

# --- 4. Monomial coefficients ---
separator("Monomial Coefficients for T_30(x)")
coeffs = T30.get_coefficients()
print(f"\n  Ascending order  c_0, c_1, ..., c_30  ({len(coeffs)} coefficients):")
for i, c in enumerate(coeffs):
    if abs(c) > 1e-40:
        print(f"    c_{i:>2d} = {mp.nstr(c, 30)}")

# --- 5. Basis generation [T_0, ..., T_8] ---
separator("Basis Generation  [T_0, ..., T_8]  dps=60")
basis = get_mpmath_chebyshev_basis(8, dps=60)
mp.mp.dps = 60
tol_basis = mp.mpf('1e-55')

print()
for poly in basis:
    v = poly.evaluate(0.75)
    expected_v = mp.cos(poly.n * mp.acos(mp.mpf('0.75')))
    diff_ok = abs(v - expected_v) < tol_basis
    status = '✓' if diff_ok else '✗'
    print(f"  T_{poly.n:>2d}(0.75) = {mp.nstr(v, 30)}   "
          f"[cos({poly.n}·arccos(0.75)) = {mp.nstr(expected_v, 30)}]   {status}")

separator("Done")