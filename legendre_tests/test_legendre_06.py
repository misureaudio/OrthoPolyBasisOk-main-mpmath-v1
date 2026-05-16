"""Test Legendre Layer 4 — HighPrecisionGaussLegendre (True Arbitrary Precision).

Demonstrates true arbitrary-precision integration where nodes, weights, and the
integrand are all mpmath mpf objects. Shows convergence of digit accuracy as n
and dps increase.
"""
from __future__ import annotations

# import math
import sys
# import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from legendre.integration import HighPrecisionGaussLegendre, gauss_legendre_high_precision


def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


separator("LAYER 4 — True Arbitrary-Precision Gauss-Legendre Integration")

# --- Reference value at very high precision ---
mp.mp.dps = 150
exact_ref = mp.exp(1) - mp.exp(-1)   # ∫_{-1}^{1} e^x dx = e - 1/e
print(f"\n  Reference value (dps=150):")
print(f"    {mp.nstr(exact_ref, 60)}")

# --- 1. Single high-precision integration ---
separator("Single Integration  n=50  dps=100")
q = HighPrecisionGaussLegendre(50, dps=100)
result = q.integrate(lambda x: mp.exp(x))

print(f"\n  Integral: ∫_{-1}^{1} e^x dx")
print(f"  n = {q.n}, dps = {q.dps}")
print(f"\n  Result ({mp.nstr(result, 98)}):")
# Print in blocks of 20 digits for readability
result_str = mp.nstr(result, 100)
for i in range(0, len(result_str), 20):
    print(f"    {result_str[i:i+20]}")

correct_digits = 0
ref_str = mp.nstr(exact_ref, 100)
for a, b in zip(result_str.replace('.', '').replace('-', ''), ref_str.replace('.', '').replace('-', '')):
    if a == b:
        correct_digits += 1
    else:
        break
print(f"\n  Correct significant digits vs reference (dps=150): {correct_digits}")

# --- 2. Node and weight inspection ---
separator("Node & Weight Inspection  n=16  dps=80")
q16 = HighPrecisionGaussLegendre(16, dps=80)

print(f"\n  {'i':>3s}   {'x_i (node)':>42s}   {'w_i (weight)':>42s}")
print(f"  {'-'*3}   {'-'*42}   {'-'*42}")
for i in range(len(q16.nodes)):
    xi = mp.nstr(q16.nodes[i], 40)
    wi = mp.nstr(q16.weights[i], 40)
    print(f"  {i:>3d}   {xi:>42s}   {wi:>42s}")

# --- 3. Convergence: increasing dps at fixed n=50 ---
separator("Precision Convergence  n = 50 (fixed)")
print(f"\n  Increasing dps shows more correct digits in the result.\n")
print(f"  {'dps':>5s}   {'Correct Digits':>16s}   {'Result (first 40 chars)':>42s}")
print(f"  {'-'*5}   {'-'*16}   {'-'*42}")

for dps in [30, 50, 80, 100, 120]:
    q_test = HighPrecisionGaussLegendre(50, dps=dps)
    val = q_test.integrate(lambda x: mp.exp(x))
    val_str = mp.nstr(val, min(dps + 5, 130))
    ref_str = mp.nstr(exact_ref, min(dps + 5, 130))

    # Count correct digits
    vd = val_str.replace('.', '').replace('-', '')
    rd = ref_str.replace('.', '').replace('-', '')
    cd = 0
    for a, b in zip(vd, rd):
        if a == b:
            cd += 1
        else:
            break

    print(f"  {dps:>5d}   {cd:>16d}   {val_str[:42]:>42s}")

# --- 4. Convergence: increasing n at fixed dps=80 ---
separator("Node Count Convergence  dps = 80 (fixed)")
print(f"\n  Increasing n (quadrature nodes) improves accuracy up to the dps limit.\n")
print(f"  {'n':>5s}   {'Correct Digits':>16s}   {'Result (first 40 chars)':>42s}")
print(f"  {'-'*5}   {'-'*16}   {'-'*42}")

for n in [8, 16, 32, 50, 80]:
    q_test = HighPrecisionGaussLegendre(n, dps=80)
    val = q_test.integrate(lambda x: mp.exp(x))
    val_str = mp.nstr(val, 78)
    ref_str = mp.nstr(exact_ref, 78)

    vd = val_str.replace('.', '').replace('-', '')
    rd = ref_str.replace('.', '').replace('-', '')
    cd = 0
    for a, b in zip(vd, rd):
        if a == b:
            cd += 1
        else:
            break

    print(f"  {n:>5d}   {cd:>16d}   {val_str[:42]:>42s}")

# --- 5. Factory function test ---
separator("Factory Function  gauss_legendre_high_precision(n=32, dps=60)")
q_factory = gauss_legendre_high_precision(32, dps=60)
result_f = q_factory.integrate(lambda x: mp.exp(x))
print(f"\n  Result:")
print(f"    {mp.nstr(result_f, 58)}")

# --- 6. Challenging integrand: oscillatory function ---
separator("Challenging Integrand  ∫_{-1}^{1} sin(20x)·e^x dx  n=80  dps=60")

# Analytical formula:
#   ∫ e^x sin(ax) dx = e^x (sin(ax) - a cos(ax)) / (1 + a²)
# Evaluated from -1 to 1:
#   [e(sin(a)-a·cos(a)) - e^{-1}(sin(-a)-a·cos(-a))] / (1+a²)
# = [2cosh(1)·sin(a) - 2a·sinh(1)·cos(a)] / (1 + a²)
mp.mp.dps = 80
a = mp.mpf(20)
ref_chal = (2*mp.cosh(1)*mp.sin(a) - 2*a*mp.sinh(1)*mp.cos(a)) / (1 + a**2)

q_chal = HighPrecisionGaussLegendre(80, dps=60)
result_chal = q_chal.integrate(lambda x: mp.sin(20 * x) * mp.exp(x))

print(f"\n  Gauss-Legendre result:")
print(f"    {mp.nstr(result_chal, 50)}")
print(f"\n  Analytical reference:")
print(f"    {mp.nstr(ref_chal, 50)}")
diff = abs(result_chal - ref_chal)
if diff > 0:
    print(f"\n  |GL - exact| = {mp.nstr(diff, 10)}")
    print(f"  Agreement to ~{int(-mp.log(diff)/mp.log(10))} decimal places")
else:
    print(f"\n  Exact match!")

separator("Done")