"""Test Legendre Layer 2 — High Precision (mpmath).

Demonstrates arbitrary-precision evaluation, differentiation, coefficient extraction,
and basis generation for Legendre polynomials.
"""
from __future__ import annotations

# import math
import sys
# import numpy as np
# import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from legendre.high_precision import LegendreMPMath, legendre_high_precision_basis


def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


separator("LAYER 2 — High Precision Legendre (mpmath)")

# --- 1. High-precision evaluation (100 decimal digits) ---
P50 = LegendreMPMath(50, dps=100)
x_val = "0.12345678901234567890"

val = P50.evaluate(x_val)
print(f"\nP_50({x_val})  [dps=100]:")
print(f"  {val}")

deriv = P50.derivative_value(x_val)
print(f"\nP_50'({x_val}) [dps=100]:")
print(f"  {deriv}")

# --- 2. Extract coefficients ---
separator("Monomial Coefficients for P_50(x)")
coeffs = P50.coefficients_ascending
print(f"\n  Ascending order  c_0, c_1, ..., c_50  ({len(coeffs)} coefficients):")
for i, c in enumerate(coeffs):
    if abs(c) > 1e-40:
        print(f"    c_{i:>2d} = {c}")

# --- 3. Basis generation ---
separator("Basis Generation  [P_0, ..., P_10]  dps=80")
basis = legendre_high_precision_basis(10, dps=80)
print()
for poly in basis:
    v = poly.evaluate(0.5)
    print(f"  P_{poly.n:>2d}(0.5) = {v}")

separator("Done")