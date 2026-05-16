"""LAYER 4 — Quadrature orchestrator."""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple, List
from .numerical import GeneralizedLaguerrePolynomial

_laguerre_quadrature_cache = {}

class LaguerreQuadrature:
    def __init__(self, n: int, alpha: float = 0.0, use_mpmath: bool = False):
        self.n, self.alpha = n, alpha
        key = (n, alpha, use_mpmath)
        if key not in _laguerre_quadrature_cache:
            if use_mpmath: _laguerre_quadrature_cache[key] = self._compute_mp(n, alpha)
            else: _laguerre_quadrature_cache[key] = self._compute_np(n, alpha)
        self._nodes, self._weights = _laguerre_quadrature_cache[key]

    @staticmethod
    def _compute_np(n, alpha):
        diag = np.array([2.0 * k + alpha + 1.0 for k in range(n)])
        off_diag = np.sqrt(np.arange(1, n) * (np.arange(1, n) + alpha))
        J = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        vals, vecs = np.linalg.eigh(J)
        idx = np.argsort(vals)
        return vals[idx], (math.gamma(alpha + 1) * vecs[0, idx]**2)

    @staticmethod
    def _compute_mp(n, alpha):
        import mpmath as mp
        with mp.workdps(80):
            a = mp.mpf(alpha)
            diag = [2*k + a + 1 for k in range(n)]
            off = [mp.sqrt(mp.mpf(k)*(k+a)) for k in range(1, n)]
            J = mp.matrix(n, n)
            for i in range(n):
                J[i,i] = diag[i]
                if i < n-1: J[i, i+1] = J[i+1, i] = off[i]
            vals, vecs = mp.eig(J, left=False)
            indices = sorted(range(n), key=lambda i: vals[i])
            nodes = [float(vals[i]) for i in indices]
            weights = [float(mp.gamma(a+1) * vecs[0,i]**2) for i in indices]
            return np.array(nodes), np.array(weights)

    def integrate(self, f: Callable) -> float:
        return float(np.sum(self._weights * np.vectorize(f)(self._nodes)))

    def integrate_mp(self, f_or_expr, dps=80):
        """Full mpmath pipeline: recompute nodes/weights at target precision,
        evaluate integrand via mpmath-compatible callable, return mp.mpf.

        Args:
            f_or_expr: A sympy expression or an mpmath-compatible callable.
            dps: Decimal places of precision (default: 80).
        Returns:
            mp.mpf — arbitrary-precision result with NO float() cast.
        """
        import mpmath as mp
        from sympy import Symbol, lambdify

        n = self.n
        a = mp.mpf(self.alpha)

        # Recompute nodes/weights at target precision (bypasses float64 cast)
        with mp.workdps(dps):
            diag = [2 * k + a + 1 for k in range(n)]
            off = [mp.sqrt(mp.mpf(k) * (k + a)) for k in range(1, n)]
            J = mp.matrix(n, n)
            for i in range(n):
                J[i, i] = diag[i]
                if i < n - 1:
                    J[i, i + 1] = J[i + 1, i] = off[i]
            vals, vecs = mp.eig(J, left=False)
            indices = sorted(range(n), key=lambda i: vals[i])

        # Build mpmath-compatible callable
        if hasattr(f_or_expr, 'atoms'):
            v = Symbol(str(list(f_or_expr.free_symbols)[0])) if f_or_expr.free_symbols else Symbol('x')
            f_mp = lambdify(v, f_or_expr, modules='mpmath')
        elif callable(f_or_expr):
            f_mp = f_or_expr
        else:
            raise TypeError("f_or_expr must be a sympy expression or an mpmath-compatible callable")

        # Summation in mpmath arithmetic — return raw mp.mpf
        with mp.workdps(dps):
            total = sum(
                mp.gamma(a + 1) * vecs[0, idx] ** 2 * f_mp(vals[idx])
                for idx in indices
            )
            return total

    @property
    def nodes(self): return self._nodes
    @property
    def weights(self): return self._weights

# --- Symbols required by __init__.py ---

class GeneralizedLaguerreBasis:
    def __init__(self, max_n: int, alpha: float = 0.0):
        self.max_n, self.alpha = max_n, alpha
        self.quad = LaguerreQuadrature(max_n + 1, alpha)
        self.polys = [GeneralizedLaguerrePolynomial(n, alpha) for n in range(max_n + 1)]

    def norm_squared(self, n: int) -> float:
        return math.gamma(n + self.alpha + 1) / math.factorial(n)

class LaguerreBasis(GeneralizedLaguerreBasis):
    def __init__(self, max_n: int):
        super().__init__(max_n, alpha=0.0)

def compute_roots(n: int, alpha: float = 0.0):
    return LaguerreQuadrature(n, alpha).nodes

def gauss_quadrature_weights(n: int, alpha: float = 0.0):
    return LaguerreQuadrature(n, alpha).weights

def function_projection(f: Callable, max_n: int, alpha: float = 0.0) -> np.ndarray:
    basis = GeneralizedLaguerreBasis(max_n, alpha)
    return np.array([basis.quad.integrate(lambda x: f(x) * basis.polys[n](x)) / 
                    basis.norm_squared(n) for n in range(max_n + 1)])

def function_approximation(f: Callable, max_n: int, alpha: float = 0.0) -> Callable:
    coeffs = function_projection(f, max_n, alpha)
    basis = GeneralizedLaguerreBasis(max_n, alpha)
    return lambda x: sum(c * p(x) for c, p in zip(coeffs, basis.polys))