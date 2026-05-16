"""LAYER 1 — Symbolic (SymPy) exact arithmetic for Hermite polynomials."""
from __future__ import annotations
import sympy as sp
from sympy import Symbol

class HermiteSymbolic:
    """Exact Hermite polynomial H_n(x) via SymPy (physicists' convention)."""

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self.n = n
        self._x = Symbol('x')
        # sp.hermite(n, x) defaults to the physicists' convention: 16x^4 - 48x^2 + 12
        self._expr = sp.hermite(self.n, self._x)

    @property
    def expression(self):
        """Returns the SymPy expression (e.g., 16*x**4 - 48*x**2 + 12)."""
        return self._expr

    def evaluate(self, x):
        """Substitute x and return the exact or symbolic result."""
        return self._expr.subs(self._x, x)

def hermite_symbolic_basis(max_n: int) -> List[HermiteSymbolic]:
    return [HermiteSymbolic(n) for n in range(max_n + 1)]