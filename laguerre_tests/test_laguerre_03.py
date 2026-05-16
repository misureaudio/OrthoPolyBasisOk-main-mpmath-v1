from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from laguerre.numerical import LaguerrePolynomial, GeneralizedLaguerrePolynomial
# import numpy as np

# ─── Standard Laguerre L_5(x) at x = 2.0 ───────────────────────────────
L5 = LaguerrePolynomial(5)
val = L5(2.0)

print("=" * 60)
print("  Standard Laguerre Polynomial  L_5(x)")
print("=" * 60)
print(f"  Degree (n) : 5")
print(f"  Parameter  : alpha = 0  (standard)")
print(f"  Point (x)  : 2.0")
print("-" * 60)
print(f"  L_5(2.0) = {val}")
print("=" * 60)

# ─── Generalized Laguerre L_10^(2)(x) on [0, 20] ──────────────────────
Lg = GeneralizedLaguerrePolynomial(10, alpha=2.0)
xs = np.linspace(0, 20, 100)
ys = Lg.evaluate(xs)  # Vectorized evaluation
coeffs = Lg.coefficients_ascending  # Monomial coefficients

print()
print("=" * 60)
print("  Generalized Laguerre Polynomial  L_10^(2)(x)")
print("=" * 60)
print(f"  Degree (n) : 10")
print(f"  Parameter  : alpha = 2.0")
print("-" * 60)
print()

# -- Sample of evaluated values (first, last, min, max) --
print("  Evaluated on [0, 20] with 100 points:")
print(f"    L_10^(2)(0)       = {ys[0]: .6e}")
print(f"    L_10^(2)(20)      = {ys[-1]: .6e}")
print(f"    min value         = {np.min(ys): .6e}  at x = {xs[np.argmin(ys)]:.4f}")
print(f"    max value         = {np.max(ys): .6e}  at x = {xs[np.argmax(ys)]:.4f}")
print()

# -- Monomial coefficients (ascending powers of x) --
print("  Monomial Coefficients (ascending powers of x):")
for i, c in enumerate(coeffs):
    print(f"    x^{i:>2}  :  {c: .6e}")

print("=" * 60)