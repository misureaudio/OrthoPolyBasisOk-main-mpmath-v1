"""Test Legendre Layer 4 — LegendreQuadrature Precision Comparison.

Compares double-precision vs mpmath-assisted node computation for large n,
demonstrating stability improvements when use_mpmath=True.
"""
from __future__ import annotations

import math
import sys
import numpy as np
# import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from legendre.integration import LegendreQuadrature


def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


separator("LAYER 4 — LegendreQuadrature Precision Comparison")

exact = 2 * math.sinh(1)

# --- 1. Double-precision quadrature (n=20) ---
separator("Double-Precision Quadrature  n = 20")
q = LegendreQuadrature(20, use_mpmath=False)
result = q.integrate(lambda x: np.exp(x))

print(f"\n  Integral: ∫_{-1}^{1} e^x dx")
print(f"  Exact value:     {exact:.15f}")
print(f"  Computed value:  {result:.15f}")
print(f"  Absolute error:  {abs(result - exact):.2e}")

# Show first/last few nodes and weights
print(f"\n  First 5 nodes:")
for i in range(5):
    print(f"    x[{i:>2d}] = {q.nodes[i]:>18.14f}   w[{i:>2d}] = {q.weights[i]:>18.14f}")
print(f"  ...")
print(f"  Last 5 nodes:")
for i in range(-5, 0):
    print(f"    x[{i:>2d}] = {q.nodes[i]:>18.14f}   w[{i:>2d}] = {q.weights[i]:>18.14f}")

# --- 2. High-precision node computation (n=100) ---
separator("High-Precision Node Computation  n = 100  dps = 80")
q_hp = LegendreQuadrature(100, use_mpmath=True, dps=80)
result_hp = q_hp.integrate(lambda x: np.exp(x))

print(f"\n  Integral: ∫_{-1}^{1} e^x dx")
print(f"  Exact value:     {exact:.15f}")
print(f"  Computed value:  {result_hp:.15f}")
print(f"  Absolute error:  {abs(result_hp - exact):.2e}")

# Show first/last few nodes and weights
print(f"\n  First 5 nodes:")
for i in range(5):
    print(f"    x[{i:>3d}] = {q_hp.nodes[i]:>18.14f}   w[{i:>3d}] = {q_hp.weights[i]:>18.14e}")
print(f"  ...")
print(f"  Last 5 nodes:")
for i in range(-5, 0):
    print(f"    x[{i:>3d}] = {q_hp.nodes[i]:>18.14f}   w[{i:>3d}] = {q_hp.weights[i]:>18.14e}")

# --- 3. Convergence study: increasing n with both modes ---
separator("Convergence Study  ∫_{-1}^{1} e^x dx")
print(f"\n  {'n':>5s}   {'use_mpmath':>10s}   {'Result':>20s}   {'Error':>14s}")
print(f"  {'-'*5}   {'-'*10}   {'-'*20}   {'-'*14}")

for n in [8, 16, 32, 64, 100]:
    for use_mp in [False, True]:
        q_test = LegendreQuadrature(n, use_mpmath=use_mp, dps=80)
        val = q_test.integrate(lambda x: np.exp(x))
        err = abs(val - exact)
        mp_label = "True" if use_mp else "False"
        print(f"  {n:>5d}   {mp_label:>10s}   {val:>20.15f}   {err:>14.2e}")

# --- 4. Weight sum verification ---
separator("Weight Sum Verification")
print(f"\n  The sum of quadrature weights should equal ∫_{-1}^{1} 1 dx = 2.0\n")
print(f"  {'n':>5s}   {'use_mpmath':>10s}   {'Sum(weights)':>22s}   {'Error from 2.0':>16s}")
print(f"  {'-'*5}   {'-'*10}   {'-'*22}   {'-'*16}")

for n in [8, 32, 64, 100]:
    for use_mp in [False, True]:
        q_test = LegendreQuadrature(n, use_mpmath=use_mp, dps=80)
        w_sum = sum(q_test.weights) if not isinstance(q_test.weights, np.ndarray) else q_test.weights.sum()
        err = abs(float(w_sum) - 2.0)
        mp_label = "True" if use_mp else "False"
        print(f"  {n:>5d}   {mp_label:>10s}   {float(w_sum):>22.16f}   {err:>16.2e}")

separator("Done")