from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from laguerre.integration import LaguerreQuadrature, function_projection
# import numpy as np

# ─── Gauss-Laguerre: ∫₀^∞ f(x) e^{-x} dx ──────────────────────────────
ql = LaguerreQuadrature(32, alpha=0.0)
result = ql.integrate(lambda x: 1 / (1 + x**2))

print("=" * 60)
print("  Gauss-Laguerre Quadrature  (alpha = 0)")
print("=" * 60)
print(f"  Integral : ∫₀^∞ [1/(1+x²)] e⁻ˣ dx")
print(f"  Points   : 32")
print(f"  alpha    : 0.0")
print(f"  Method   : NumPy (eigenvalue)")
print("-" * 60)
print(f"  Result   : {result: .12e}")
print(f"  Reference: π/(e²) ≈ 0.8913... (analytical)")
print("=" * 60)

# ─── Generalized: ∫₀^∞ f(x) x^α e^{-x} dx ─────────────────────────────
ql_gen = LaguerreQuadrature(50, alpha=1.5, use_mpmath=True)
result_gen = ql_gen.integrate(lambda x: np.exp(-x/3))

print()
print("=" * 60)
print("  Generalized Gauss-Laguerre Quadrature  (alpha = 1.5)")
print("=" * 60)
print(f"  Integral : ∫₀^∞ [e^(-x/3)] x^(1.5) e⁻ˣ dx")
print(f"  Points   : 50")
print(f"  alpha    : 1.5")
print(f"  Method   : mpmath (high-precision eigenvalue)")
print("-" * 60)
print(f"  Result   : {result_gen: .12e}")
print("=" * 60)

# ─── Function projection (spectral expansion) ──────────────────────────
f = lambda x: np.exp(-2*x) * np.cos(x)
coeffs = function_projection(f, max_n=15, alpha=0.0)

print()
print("=" * 60)
print("  Function Projection (Spectral Expansion)")
print("=" * 60)
print(f"  Function : f(x) = e^(-2x) · cos(x)")
print(f"  Basis    : Standard Laguerre  (alpha = 0)")
print(f"  Max n    : 15  (16 coefficients)")
print("-" * 60)
print()
print("  Spectral Coefficients c_n:")
for i, c in enumerate(coeffs):
    bar_len = min(int(abs(c) * 500), 40)
    bar = "█" * bar_len + ("→" if c >= 0 else "←")
    sign = "+" if c >= 0 else "-"
    print(f"    c_{i:>2} = {sign}{abs(c):.6e}  {bar}")
print()
print(f"  ‖c‖₂ = {np.linalg.norm(coeffs): .6e}")
print("=" * 60)