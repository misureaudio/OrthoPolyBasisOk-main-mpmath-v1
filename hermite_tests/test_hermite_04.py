from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from hermite.integration import GaussHermiteQuadrature, HermiteProjection
# import numpy as np
# import math

# ── Gauss-Hermite Quadrature ─────────────────────────────────────────────────
# ∫_{-∞}^{∞} f(x) e^{-x²} dx
qh = GaussHermiteQuadrature(32)
result = qh.integrate(lambda x: np.exp(-x**2 / 4))
# Exact: ∫ exp(-x²/4)·exp(-x²) dx = ∫ exp(-5x²/4) dx = sqrt(4π/5)
exact = math.sqrt(4 * math.pi / 5)

print("=" * 62)
print("  Gauss-Hermite Quadrature  (32-point)")
print("  ∫_{-∞}^{∞} exp(-x²/4) · e^{-x²} dx")
print("=" * 62)
print(f"  Quadrature result : {result:.15e}")
print(f"  Exact value       : {exact:.15e}")
print(f"  Absolute error    : {abs(result - exact):.6e}")
print("=" * 62)

# ── Hermite Projection (Spectral Expansion) ──────────────────────────────────
proj = HermiteProjection(max_degree=10)
f = lambda x: np.exp(-x**2 / 2)
coeffs = proj.project(f)  # Hermite coefficients
xs = np.linspace(-5, 5, 200)
approx = proj.approximate(xs, coeffs)  # Reconstruct

print("\n" + "=" * 62)
print("  Hermite Projection  —  f(x) = exp(-x²/2)")
print("  Max degree = 10   |   Quadrature points = 11")
print("=" * 62)
print("\n  Hermite Coefficients:")
print("  " + "-" * 40)
print(f"  {'k':>4s}  {'cₖ':>18s}  {'|cₖ|':>12s}")
print("  " + "-" * 40)
for k, c in enumerate(coeffs):
    print(f"  {k:4d}  {c:>18.12e}  {abs(c):12.6e}")
print("  " + "-" * 40)

# Approximation quality
f_exact = f(xs)
max_err = np.max(np.abs(approx - f_exact))
rms_err = np.sqrt(np.mean((approx - f_exact)**2))

print(f"\n  Approximation Quality  (on x ∈ [-5, 5]):")
print(f"    Max absolute error : {max_err:.6e}")
print(f"    RMS error          : {rms_err:.6e}")

# Sample table of approximation vs exact
step = max(1, len(xs) // 16)
print("\n  " + "-" * 54)
print(f"  {'i':>3s}  {'x':>10s}  {'f(x)':>14s}  {'approx(x)':>14s}")
print("  " + "-" * 54)
for i in range(0, len(xs), step):
    print(f"  {i:3d}  {xs[i]:10.6f}  {f_exact[i]:14.8f}  {approx[i]:14.8f}")
print("  " + "-" * 54)

# ── Hermite Transform Pair ───────────────────────────────────────────────────
from hermite.integration import hermite_transform, inverse_hermite_transform
c = hermite_transform(lambda x: np.cos(x), 8)
reconstructed = inverse_hermite_transform(c, xs)

print("\n" + "=" * 62)
print("  Hermite Transform Pair  —  g(x) = cos(x)")
print("  Forward transform degree = 8")
print("=" * 62)
print("\n  Forward Transform Coefficients:")
print("  " + "-" * 40)
print(f"  {'k':>4s}  {'ĝₖ':>18s}")
print("  " + "-" * 40)
for k, ck in enumerate(c):
    print(f"  {k:4d}  {ck:>18.12e}")
print("  " + "-" * 40)

# Reconstructed vs original
g_exact = np.cos(xs)
max_err_cos = np.max(np.abs(reconstructed - g_exact))

print(f"\n  Reconstruction Error:")
print(f"    Max absolute error : {max_err_cos:.6e}")

step2 = max(1, len(xs) // 16)
print("\n  " + "-" * 54)
print(f"  {'i':>3s}  {'x':>10s}  {'cos(x)':>14s}  {'reconstructed':>14s}")
print("  " + "-" * 54)
for i in range(0, len(xs), step2):
    print(f"  {i:3d}  {xs[i]:10.6f}  {g_exact[i]:14.8f}  {reconstructed[i]:14.8f}")
print("  " + "-" * 54)