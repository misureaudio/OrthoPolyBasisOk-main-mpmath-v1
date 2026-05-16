"""Chebyshev-T Layer 3 — Numerical (NumPy) Pretty-Printed Tests."""

from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from chebyshev.numerical import (
    ChebyshevPolynomial,
    ChebyshevGenerator,
    chebyshev_coefficients,
    chebyshev_derivative_stable,
    chebyshev_integral_stable,
    generate_numpy_chebyshev,
    get_numpy_chebyshev_basis,
)
# import numpy as np

# ── Header ───────────────────────────────────────────────────────────
print("=" * 72)
print("  Chebyshev-T Layer 3 — Numerical (NumPy)")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════════
# 1. Monomial Coefficients via ChebyshevPolynomial.coefficients
# ══════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("  T_n(x) monomial coefficients (ascending order)")
print("-" * 72)

for n in range(8):
    Tn = ChebyshevPolynomial(n)
    coeffs = Tn.coefficients  # ascending: c_0, c_1, ..., c_n
    # Format as polynomial string
    terms = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-15:
            continue
        sign = "+" if c > 0 else "−"
        ac = abs(c)
        if i == 0:
            term = f"{sign} {ac:.6f}"
        elif i == 1:
            term = f"{sign} {ac:.6f} x"
        else:
            term = f"{sign} {ac:.6f} x^{i}"
        terms.append(term)
    poly_str = " ".join(terms) if terms else "0"
    # Remove leading "+ "
    if poly_str.startswith("+ "):
        poly_str = poly_str[2:]
    print(f"  T_{n:>2d}(x) = {poly_str}")

# ══════════════════════════════════════════════════════════════════════
# 2. Vectorized Evaluation — Statistics at Sample Points
# ══════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("  T_n(x) evaluation table (n = 0..10, sample x values)")
print("-" * 72)

xs = np.array([-1.0, -0.5, 0.0, 0.3, 0.7, 0.99, 1.0])
print()
# Header row
header = f"{'n':>4s}  " + "  ".join(f"x={x:>6.2f}".rjust(14) for x in xs)
print(header)
print("  " + "-" * (4 + 14 * len(xs)))

for n in range(11):
    Tn = ChebyshevPolynomial(n)
    vals = [Tn(x) for x in xs]
    row = f"{'%02d' % n}  " + "  ".join(f"{v:>14.8f}" for v in vals)
    print(row)

# ══════════════════════════════════════════════════════════════════════
# 3. Derivative & Integral via Clenshaw (Stable)
# ══════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("  T'_n(x) and ∫T_n(x) dx at sample points (Clenshaw, stable)")
print("-" * 72)

test_ns = [5, 10, 15, 20]
test_xs = [-0.8, -0.3, 0.0, 0.4, 0.9]

for n in test_ns:
    Tn = ChebyshevPolynomial(n)
    print(f"\n  n = {n}:")
    # print(f"  {'x':>8s}  {'T_n(x)':>14s}  {'T\'_n(x)':>14s}  {'integral':>14s}")
    print(f"  {'x':>8s}  {'T_n(x)':>14s}  {'T''_n(x)':>14s}  {'integral':>14s}")
    print("  " + "-" * 56)
    for x in test_xs:
        val = Tn(x)
        deriv = Tn.derivative(x)
        integ = Tn.integral(x)
        print(f"  {x:>8.2f}  {val:>14.8f}  {deriv:>14.8f}  {integ:>14.8f}")

# ══════════════════════════════════════════════════════════════════════
# 4. Functional API — chebyshev_derivative_stable / integral
# ══════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("  Functional API: derivative & integral (n=20, x=0.3)")
print("-" * 72)

n, x = 20, 0.3
d = chebyshev_derivative_stable(n, x)
i = chebyshev_integral_stable(n, x)
Tn = ChebyshevPolynomial(n)
print(f"  T_{n}({x})       = {Tn(x):.12f}")
print(f"  T'_{n}({x})      = {d:.12f}")
print(f"  integral T_{n}   = {i:.12f}")

# ══════════════════════════════════════════════════════════════════════
# 5. Cached Generator — Demonstrate Cache Hits
# ══════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("  ChebyshevGenerator cache behavior")
print("-" * 72)

gen = ChebyshevGenerator()
print(f"  Initial cache size: {len(gen._cache)}")

# First call — misses cache
c1 = gen.get_monomial_coefficients(10)
print(f"  After get_monomial_coefficients(10): cache size = {len(gen._cache)}")

# Second call — hits cache (same reference check)
c2 = gen.get_monomial_coefficients(10)
print(f"  Second call (should hit cache):   cache size = {len(gen._cache)}")
print(f"  Cache hit confirmed: {c1 is not c2 and np.allclose(c1, c2)} "
      "(different objects, same values)")

# Request several more
for nn in [5, 8, 15, 20]:
    gen.get_monomial_coefficients(nn)
print(f"  After requesting n=5,8,15,20:     cache size = {len(gen._cache)}")

# ══════════════════════════════════════════════════════════════════════
# 6. generate_numpy_chebyshev / get_numpy_chebyshev_basis
# ══════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("  NumPy array API: generate_numpy_chebyshev & basis")
print("-" * 72)

arr = generate_numpy_chebyshev(6)
print(f"\n  T_6 coefficients (numpy array, ascending):")
print(f"    {arr}")
print(f"    dtype: {arr.dtype}, shape: {arr.shape}")

basis = get_numpy_chebyshev_basis(5)
print(f"\n  Basis T_0 .. T_5 (list of numpy arrays):")
for n, b in enumerate(basis):
    print(f"    T_{n}: {b}")

# ══════════════════════════════════════════════════════════════════════
# 7. Verify T_n(cos(theta)) = cos(n*theta) at Numerical Precision
# ══════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("  Identity check: T_n(cos θ) ≈ cos(n·θ)  (max |error|)")
print("-" * 72)

thetas = np.linspace(0, np.pi, 100)
for n in [5, 10, 20, 30, 50]:
    Tn = ChebyshevPolynomial(n)
    xs_cos = np.cos(thetas)
    vals_direct = np.array([Tn(x) for x in xs_cos])
    vals_trig = np.cos(n * thetas)
    max_err = np.max(np.abs(vals_direct - vals_trig))
    print(f"  n={n:>3d}:  max |T_n(cos θ) − cos(nθ)| = {max_err:.2e}")

# ── Footer ───────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  Done.")
print("=" * 72)