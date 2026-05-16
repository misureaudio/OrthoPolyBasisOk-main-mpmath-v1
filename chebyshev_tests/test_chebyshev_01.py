"""Test Chebyshev Layer 1 — Symbolic (SymPy).

Demonstrates exact symbolic expressions, coefficient extraction, evaluation,
derivative, and basis generation for Chebyshev-T polynomials.
"""

from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

import sympy as sp
from chebyshev.symbolic import ChebyshevSymbolic, chebyshev_symbolic_basis


def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


separator("LAYER 1 — Symbolic Chebyshev-T (SymPy)")

# --- Single polynomial T_5(x) ---
T5 = ChebyshevSymbolic(5)

print(f"\nT_5(x) expression:")
sp.pprint(T5.expression)

# Evaluate at exact rational point
val_half = T5.evaluate(sp.Rational(1, 2))
print(f"\nT_5(1/2) = {val_half}   (exact: cos(5·π/3) = 1/2)")

# --- Derivative and integral ---
separator("Derivative & Integral")
dT5 = sp.diff(T5.expression, sp.Symbol('x'))
print(f"\nT_5'(x):")
sp.pprint(dT5)

iT5 = sp.integrate(T5.expression, sp.Symbol('x'))
print(f"\n∫T_5(x) dx:")
sp.pprint(iT5)

# --- Basis generation [T_0, ..., T_6] ---
separator("Basis Generation  [T_0, ..., T_6]")
basis = chebyshev_symbolic_basis(6)
x = sp.Symbol('x')
print()
for poly in basis:
    print(f"  T_{poly.n}({sp.sstr(x)}) = {sp.srepr(poly.expression)}")

# --- Verify T_n(cos θ) = cos(nθ) identity ---
separator("Identity Verification  T_n(cos θ) = cos(n·θ)")
theta = sp.Symbol('theta')
print()
for poly in basis[:5]:
    # Substitute x = cos(theta) and simplify
    substituted = poly.expression.subs(x, sp.cos(theta))
    simplified = sp.trigsimp(substituted)
    expected = sp.cos(poly.n * theta)
    match = sp.simplify(simplified - expected) == 0
    print(f"  T_{poly.n}(cos θ) = cos({poly.n}·θ):  {'✓' if match else '✗'}")

separator("Done")