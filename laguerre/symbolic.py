"""LAYER 1 — Symbolic (SymPy) exact arithmetic for Laguerre polynomials."""
from __future__ import annotations
from typing import Optional, List

try:
    import sympy as sp
    from sympy import S, Rational, Poly, Symbol
except ImportError:
    raise ImportError("sympy is required. Install with: pip install sympy")


class LaguerreSymbolic:
    """Exact Laguerre polynomial L_n^(α)(x) via SymPy."""

    def __init__(self, n: int, alpha: float = 0.0):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if alpha <= -1:
            raise ValueError("alpha must be > -1")
        self.n = n
        self.alpha = alpha
        self._x = Symbol('x')
        self._expr = None

    @property
    def expression(self):
        if self._expr is None:
            if self.alpha == 0:
                self._expr = sp.laguerre(self.n, self._x)
            else:
                self._expr = sp.assoc_laguerre(self.n, self.alpha, self._x)
        return self._expr

    def evaluate(self, x):
        return self.expression.subs(self._x, x)


def laguerre_symbolic_basis(max_n: int, alpha: float = 0.0) -> List[LaguerreSymbolic]:
    return [LaguerreSymbolic(n, alpha) for n in range(max_n + 1)]