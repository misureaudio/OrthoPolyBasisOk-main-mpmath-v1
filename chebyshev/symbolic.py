"""LAYER 1 — Symbolic (SymPy) exact arithmetic for Chebyshev polynomials."""
from __future__ import annotations
from typing import Optional, List

try:
    import sympy as sp
    from sympy import S, Rational, Poly, Symbol
except ImportError:
    raise ImportError("sympy is required. Install with: pip install sympy")


class ChebyshevSymbolic:
    """Exact Chebyshev T_n polynomial via SymPy."""

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self.n = n
        self._x = Symbol('x')
        self._expr = None

    @property
    def expression(self):
        if self._expr is None:
            self._expr = sp.chebyshevt(self.n, self._x)
        return self._expr

    def evaluate(self, x):
        return self.expression.subs(self._x, x)


def chebyshev_symbolic_basis(max_n: int) -> List[ChebyshevSymbolic]:
    return [ChebyshevSymbolic(n) for n in range(max_n + 1)]


def generate_sympy_chebyshev(n: int):
    return ChebyshevSymbolic(n).expression


def get_sympy_chebyshev_basis(max_n: int):
    return [ChebyshevSymbolic(n).expression for n in range(max_n + 1)]