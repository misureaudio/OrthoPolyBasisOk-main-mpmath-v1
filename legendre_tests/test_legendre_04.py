"""Test Legendre Layer 4 — Quadrature (Integration).

Demonstrates Gauss-Legendre quadrature via Golub-Welsch and Newton-Raphson,
precision-aware LegendreQuadrature, HighPrecisionGaussLegendre, and convenience functions.
"""
from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from legendre.integration import (
    GaussLegendreQuadrature,
    LegendreQuadrature,
    HighPrecisionGaussLegendre,
    gauss_legendre,
    gauss_legendre_newton,
    gauss_legendre_golub_welsch,
    gauss_legendre_high_precision,
)


def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


separator("LAYER 4 — Gauss-Legendre Quadrature")

# --- 1. Golub-Welsch nodes and weights (n=10) ---
nodes, weights = GaussLegendreQuadrature.golub_welsch(10)

separator("Golub-Welsch  n = 10  nodes & weights")
print(f"\n  {'i':>3s}   {'x_i (node)':>20s}   {'w_i (weight)':>20s}")
print(f"  {'-'*3}   {'-'*20}   {'-'*20}")
for i in range(len(nodes)):
    print(f"  {i:>3d}   {nodes[i]:20.15f}   {weights[i]:20.15f}")
print(f"\n  Sum of weights = {weights.sum():.15f}  (expected: 2.0)")

# --- 2. Newton-Raphson comparison ---
nr_nodes, nr_weights = GaussLegendreQuadrature.newton_raphson(10)

separator("Newton-Raphson vs Golub-Welsch  (n=10)")
print(f"\n  {'i':>3s}   {'|GW - NR| node':>18s}   {'|GW - NR| weight':>18s}")
print(f"  {'-'*3}   {'-'*18}   {'-'*18}")
max_node_err = 0.0
max_wgt_err = 0.0
for i in range(10):
    ne = abs(nodes[i] - nr_nodes[i])
    we = abs(weights[i] - nr_weights[i])
    max_node_err = max(max_node_err, ne)
    max_wgt_err = max(max_wgt_err, we)
    print(f"  {i:>3d}   {ne:18.2e}   {we:18.2e}")
print(f"\n  Max node error:   {max_node_err:.2e}")
print(f"  Max weight error: {max_wgt_err:.2e}")

# --- 3. Integration test: ∫_{-1}^{1} e^x dx = 2·sinh(1) ---
separator("Integration Test  ∫_{-1}^{1} e^x dx")
exact = 2 * math.sinh(1)
print(f"\n  Exact value:  {exact:.15f}")

test_func = lambda x: np.exp(x)
for n in [4, 8, 16, 32]:
    q = LegendreQuadrature(n, use_mpmath=False)
    result = q.integrate(test_func)
    err = abs(result - exact)
    print(f"  n = {n:>3d}:  result = {result:.15f}   error = {err:.2e}")

# --- 4. High-precision quadrature ---
separator("High-Precision Quadrature  ∫_{-1}^{1} e^x dx")
mp_func = lambda x: mp.exp(x)

for dps in [50, 80, 100]:
    q_hp = HighPrecisionGaussLegendre(32, dps=dps)
    result_mp = q_hp.integrate(mp_func)
    # Use mp.nstr to display full precision
    digits = max(15, dps - 2)
    print(f"  dps={dps:>4d}:  {mp.nstr(result_mp, digits)}")

# --- 5. Convenience function comparison ---
separator("Convenience Functions  (n=8)")
gw_nodes, gw_wts = gauss_legendre(8)
nr_nodes2, nr_wts2 = gauss_legendre_newton(8)
gw2_nodes, gw2_wts = gauss_legendre_golub_welsch(8)

print(f"\n  gauss_legendre == gauss_legendre_golub_welsch:  {np.allclose(gw_nodes, gw2_nodes)}")
print(f"  gauss_legendre ~= gauss_legendre_newton:        {np.allclose(gw_nodes, nr_nodes2)}")

# --- 6. Factory function ---
q_factory = gauss_legendre_high_precision(16, dps=60)
result_factory = q_factory.integrate(mp_func)
separator("Factory Function  gauss_legendre_high_precision(n=16, dps=60)")
print(f"\n  Result: {mp.nstr(result_factory, 58)}")

separator("Done")