from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from hermite.numerical import HermitePolynomial
# import numpy as np

# ── Scalar Evaluation ────────────────────────────────────────────────────────
H10 = HermitePolynomial(10)
val = H10(1.5)  # Scalar evaluation

print("=" * 60)
print("  Hermite Polynomial  H₁₀(x)  —  Scalar Evaluation")
print("=" * 60)
print(f"  H₁₀(1.5) = {val:.12f}")
print("=" * 60)

# ── Monomial Coefficients ────────────────────────────────────────────────────
coeffs = H10.coefficients_ascending

print("\n  Monomial Coefficients  [c₀, c₁, ..., c₁₀]:")
print("  " + "-" * 46)
print(f"  {'Power':>6s}  {'Coefficient':>18s}")
print("  " + "-" * 46)
for i, c in enumerate(coeffs):
    print(f"  x^{i:2d}   {c:>18.6f}")
print("  " + "-" * 46)

# ── Vectorized Evaluation on [-5, 5] ─────────────────────────────────────────
xs = np.linspace(-5, 5, 200)
ys = HermitePolynomial.evaluate(xs, 10)  # Vectorized

print("\n" + "=" * 60)
print("  H₁₀(x)  —  Vectorized Evaluation  on  x ∈ [-5, 5]")
print("=" * 60)
print(f"  Samples : {len(xs)}")
print(f"  Min(H₁₀): {ys.min():.6f}")
print(f"  Max(H₁₀): {ys.max():.6f}")

# Print a subset of (x, H₁₀(x)) values at regular intervals
step = max(1, len(xs) // 21)
print("\n  " + "-" * 42)
print(f"  {'i':>4s}  {'x':>12s}  {'H₁₀(x)':>18s}")
print("  " + "-" * 42)
for i in range(0, len(xs), step):
    print(f"  {i:4d}  {xs[i]:12.6f}  {ys[i]:18.6f}")
print("  " + "-" * 42)

# ── Roots (sign changes) ─────────────────────────────────────────────────────
root_indices = np.where(np.sign(ys[:-1]) * np.sign(ys[1:]) <= 0)[0]
roots_approx = xs[root_indices]

print("\n" + "=" * 60)
print(f"  Approximate Roots of H₁₀(x)  ({len(roots_approx)} found)")
print("=" * 60)
print(f"  {'k':>2s}  {'xₖ (approx)':>14s}")
print("  " + "-" * 20)
for k, r in enumerate(roots_approx):
    print(f"  {k+1:2d}  {r:>14.8f}")
print("  " + "-" * 20)