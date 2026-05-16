"""LAYER 1 — Symbolic (SymPy) exact arithmetic for Legendre polynomials.

Provides rational coefficients, symbolic expressions, and ground-truth evaluation
for verification of the numerical layers.
"""
from __future__ import annotations
from typing import Optional, List

try:
    import sympy as sp
    from sympy import S, Rational, Poly, Symbol, Function
except ImportError:
    raise ImportError("sympy is required for symbolic layer. Install with: pip install sympy")


class LegendreSymbolic:
    """Exact Legendre polynomial via SymPy."""

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self.n = n
        self._x = Symbol('x')
        self._expr: Optional[sp.Basic] = None
        self._coeffs_ascending: Optional[List[Rational]] = None

    @property
    def expression(self) -> sp.Basic:
        if self._expr is None:
            self._expr = sp.legendre(self.n, self._x)
        return self._expr

    @property
    def coefficients_ascending(self) -> List[Rational]:
        """Coefficients [c_0, c_1, ..., c_n] for sum c_i * x^i."""
        if self._coeffs_ascending is None:
            p = Poly(self.expression, self._x)
            all_coeffs = [S.Zero] * (self.n + 1)
            # Use .nth(i) to get coefficient of x^i — compatible with all SymPy versions
            for i in range(self.n + 1):
                all_coeffs[i] = p.nth(i)
            self._coeffs_ascending = list(all_coeffs)
        return self._coeffs_ascending

    @property
    def coefficients_descending(self) -> List[Rational]:
        """Coefficients [c_n, c_{n-1}, ..., c_0] for sum c_i * x^{n-i}."""
        return list(reversed(self.coefficients_ascending))

    def evaluate(self, x):
        """Evaluate at exact or numeric point."""
        return self.expression.subs(self._x, x)

    def derivative(self) -> sp.Basic:
        return sp.diff(self.expression, self._x)

    def integral(self) -> sp.Basic:
        return sp.integrate(self.expression, self._x)


def legendre_symbolic_basis(max_n: int) -> List[LegendreSymbolic]:
    """Return [L_0, L_1, ..., L_{max_n}] as LegendreSymbolic instances."""
    return [LegendreSymbolic(n) for n in range(max_n + 1)]


# Convenience functions matching old API
def generate_sympy_legendre(n: int) -> sp.Basic:
    """Return SymPy expression for P_n(x)."""
    return LegendreSymbolic(n).expression


def get_sympy_legendre_basis(max_n: int) -> List[sp.Basic]:
    """Return [P_0, P_1, ..., P_{max_n}] as SymPy expressions."""
    return [LegendreSymbolic(n).expression for n in range(max_n + 1)]