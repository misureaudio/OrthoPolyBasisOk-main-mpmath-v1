"""Test Legendre Layer 3 — Numerical (NumPy).

Demonstrates fast evaluation, derivative/integral coefficients, vectorized batch
processing, and cached coefficient generation for Legendre polynomials.
"""
from __future__ import annotations

# import math
import sys
import numpy as np
# import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from legendre.numerical import LegendrePolynomial, LegendreGenerator


def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


separator("LAYER 3 — Numerical Legendre (NumPy)")

# --- 1. Instance-based API ---
P7 = LegendrePolynomial(7)
val = P7(0.5)

print(f"\nP_7(x) evaluation at x = 0.5:")
print(f"  P_7(0.5) = {val:.15f}")

# Derivative coefficients
deriv_coeffs = P7.derivative_coefficients()
separator("Derivative Coefficients  P_7'(x)")
print(f"\n  Descending order (degree {len(deriv_coeffs) - 1}):")
for i, c in enumerate(deriv_coeffs):
    power = len(deriv_coeffs) - 1 - i
    sign = " + " if c >= 0 else " - "
    print(f"    {sign}{abs(c):>16.12f} · x^{power}")

# Integral coefficients
integ_coeffs = P7.integral_coefficients()
separator("Integral Coefficients  ∫P_7(x) dx")
print(f"\n  Descending order (degree {len(integ_coeffs) - 1}):")
for i, c in enumerate(integ_coeffs):
    power = len(integ_coeffs) - 1 - i
    if abs(c) > 1e-15:
        sign = " + " if c >= 0 else " - "
        print(f"    {sign}{abs(c):>16.12f} · x^{power}")

# --- 2. Static vectorized evaluation ---
separator("Vectorized Evaluation  P_20(x) on [-1, 1]")
xs = np.linspace(-1, 1, 1000)
ys = LegendrePolynomial.evaluate(xs, 20)

print(f"\n  Array shape: {ys.shape}  ({len(xs)} points)")
print(f"  P_20(-1.0) = {ys[0]:>20.15f}")
print(f"  P_20( 0.0) = {ys[len(xs)//2]:>20.15f}")
print(f"  P_20(+1.0) = {ys[-1]:>20.15f}")
print(f"  min(P_20)  = {ys.min():>20.15f}  at x = {xs[ys.argmin()]:.6f}")
print(f"  max(P_20)  = {ys.max():>20.15f}  at x = {xs[ys.argmax()]:.6f}")

# Sample values at evenly spaced points
separator("Sample Values  P_20(x)")
sample_indices = np.linspace(0, len(xs)-1, 21, dtype=int)
print(f"\n  {'x':>10s}   {'P_20(x)':>20s}")
print(f"  {'-'*10}   {'-'*20}")
for idx in sample_indices:
    print(f"  {xs[idx]:10.6f}   {ys[idx]:20.15f}")

# --- 3. Cached generator ---
separator("Cached Coefficient Generator")
gen = LegendreGenerator()

print("\n  Descending coefficients for P_30(x):")
coeffs = gen.get_coefficients_descending(30)
for i, c in enumerate(coeffs):
    power = len(coeffs) - 1 - i
    if abs(c) > 1e-6:
        print(f"    x^{power:>2d}: {c:>+24.16e}")

separator("Done")