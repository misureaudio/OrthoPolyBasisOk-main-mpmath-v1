from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from chebyshev.integration import clencurt, ChebyshevQuadrature
# import numpy as np

# ── Clenshaw-Curtis Quadrature ──────────────────────────────────────────────
N = 32
nodes, weights = clencurt(N)  # (N+1)-point Clenshaw-Curtis quadrature
result = np.sum(weights * np.exp(nodes))  # ∫_{-1}^{1} e^x dx
exact = np.e - 1 / np.e  # 2*sinh(1)

print("=" * 60)
print("  Clenshaw-Curtis Quadrature  (∫_{-1}^{1} eˣ dx)")
print("=" * 60)
print(f"  Number of points : {len(nodes)}")
print(f"  Quadrature result: {result:.15e}")
print(f"  Exact value      : {exact:.15e}")
print(f"  Absolute error   : {abs(result - exact):.6e}")
print("=" * 60)

print("\n  Nodes and Weights:")
print("  " + "-" * 58)
print(f"  {'i':>4s}  {'xᵢ (node)':>18s}  {'wᵢ (weight)':>18s}")
print("  " + "-" * 58)
for i in range(len(nodes)):
    print(f"  {i:4d}  {nodes[i]:18.12f}  {weights[i]:18.12f}")
print("  " + "-" * 58)

# ── Chebyshev Extrema Points ────────────────────────────────────────────────
cq = ChebyshevQuadrature()
M = 16
extrema = cq.get_extrema_points(M)  # cos(kπ/M), k=0..M

print("\n" + "=" * 60)
print(f"  Chebyshev Extrema Points  (cos(kπ/{M}), k=0..{M})")
print("=" * 60)
print(f"  {'k':>4s}  {'xₖ = cos(kπ/M)':>18s}  {'kπ/M (rad)':>14s}")
print("  " + "-" * 50)
for k in range(len(extrema)):
    angle = k * np.pi / M
    print(f"  {k:4d}  {extrema[k]:18.12f}  {angle:14.10f}")
print("  " + "-" * 50)
print("=" * 60)
