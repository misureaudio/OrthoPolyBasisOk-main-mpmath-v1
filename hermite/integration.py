"""LAYER 4 — Quadrature and Projection orchestrator for Hermite polynomials."""
from __future__ import annotations
from typing import Callable, Tuple, List
import numpy as np
import math
import warnings
import mpmath as mp

# Orchestration: Import lower layers
from .numerical import HermitePolynomial
from .high_precision import HermiteMPMath

_hermite_quadrature_cache = {}

class GaussHermiteQuadrature:
    """Gauss-Hermite quadrature with precision-aware weight computation."""

    def __init__(self, n: int, use_mpmath: bool = False, dps: int = 80):
        self.n = n
        self._use_mpmath = use_mpmath
        self._dps = dps

        cache_key = (n, use_mpmath, dps)
        if cache_key not in _hermite_quadrature_cache:
            if use_mpmath:
                _hermite_quadrature_cache[cache_key] = self._compute_golub_welsch_mp(n, dps)
            else:
                _hermite_quadrature_cache[cache_key] = self._compute_golub_welsch_np(n)

        self._nodes, self._weights = _hermite_quadrature_cache[cache_key]

    @staticmethod
    def _compute_golub_welsch_np(n: int) -> Tuple[np.ndarray, np.ndarray]:
        diag = np.zeros(n)
        off_diag = np.sqrt(np.arange(1, n) / 2.0)
        J = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        eigenvalues, eigenvectors = np.linalg.eigh(J)
        weights = math.sqrt(math.pi) * (eigenvectors[0, :] ** 2)
        return eigenvalues, weights

    @staticmethod
    def _compute_golub_welsch_mp(n: int, dps: int) -> Tuple[np.ndarray, np.ndarray]:
        with mp.workdps(dps):
            diag = [mp.mpf(0)] * n
            off_diag = [mp.sqrt(mp.mpf(k) / 2) for k in range(1, n)]
            J = mp.matrix(n, n)
            for i in range(n):
                J[i, i] = diag[i]
                if i < n - 1:
                    J[i, i + 1] = J[i + 1, i] = off_diag[i]
            eigenvals, eigenvecs = mp.eig(J)
            # Maintain mpmath precision for as long as possible
            nodes = eigenvals
            weights = [mp.sqrt(mp.pi) * (eigenvecs[0, i] ** 2) for i in range(n)]
            # Sort by nodes
            combined = sorted(zip(nodes, weights), key=lambda x: x[0])
            n_sorted, w_sorted = zip(*combined)
            return np.array([float(x) for x in n_sorted]), np.array([float(w) for w in w_sorted])

    def integrate(self, f: Callable) -> float:
        return float(np.sum(self._weights * f(self._nodes)))

    def integrate_mp(self, f_or_expr, dps=None):
        """Full mpmath pipeline: recompute nodes/weights at target precision,
        evaluate integrand via mpmath-compatible callable, return mp.mpf.

        Args:
            f_or_expr: A sympy expression or an mpmath-compatible callable.
            dps: Decimal places of precision (default: self._dps).
        Returns:
            mp.mpf — arbitrary-precision result with NO float() cast.
        """
        import mpmath as mp
        from sympy import Symbol, lambdify

        target_dps = dps if dps is not None else self._dps
        n = self.n

        # Recompute nodes/weights at target precision (bypasses float64 cast)
        with mp.workdps(target_dps):
            diag = [mp.mpf(0)] * n
            off_diag = [mp.sqrt(mp.mpf(k) / 2) for k in range(1, n)]
            J = mp.matrix(n, n)
            for i in range(n):
                J[i, i] = diag[i]
                if i < n - 1:
                    J[i, i + 1] = J[i + 1, i] = off_diag[i]
            eigenvals, eigenvecs = mp.eig(J)

            nodes_mp = list(eigenvals)
            weights_mp = [mp.sqrt(mp.pi) * (eigenvecs[0, i] ** 2) for i in range(n)]
            combined = sorted(zip(nodes_mp, weights_mp), key=lambda x: x[0])
            nodes_mp, weights_mp = zip(*combined)

        # Build mpmath-compatible callable
        if hasattr(f_or_expr, 'atoms'):
            # sympy expression — lambdify with mpmath modules
            v = Symbol(str(list(f_or_expr.free_symbols)[0])) if f_or_expr.free_symbols else Symbol('x')
            f_mp = lambdify(v, f_or_expr, modules='mpmath')
        elif callable(f_or_expr):
            f_mp = f_or_expr
        else:
            raise TypeError("f_or_expr must be a sympy expression or an mpmath-compatible callable")

        # Summation in mpmath arithmetic — return raw mp.mpf
        with mp.workdps(target_dps):
            total = sum(w * f_mp(x) for x, w in zip(nodes_mp, weights_mp))
            return total

class HermiteProjection:
    """Project functions using the corrected physicists' normalization (H2)."""

    def __init__(self, max_degree: int, use_mpmath: bool = False):
        self.max_degree = max_degree
        self.use_mpmath = use_mpmath
        # FIX H2: Norm^2 = sqrt(pi) * 2^k * k!
        self.norm_sq = np.array([
            math.sqrt(math.pi) * (2**k) * math.factorial(k) 
            for k in range(max_degree + 1)
        ])
        self.quad = GaussHermiteQuadrature(max_degree + 1, use_mpmath=use_mpmath)

    def project(self, f: Callable) -> np.ndarray:
        coeffs = []
        for k in range(self.max_degree + 1):
            # Orchestrate Layer 2 or 3 based on preference
            poly = HermiteMPMath(k) if self.use_mpmath else HermitePolynomial(k)
            integrand = lambda x: f(x) * poly(x)
            coeffs.append(self.quad.integrate(integrand) / self.norm_sq[k])
        return np.array(coeffs)

    def approximate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x, dtype=np.float64)
        for k, c in enumerate(coeffs):
            if abs(c) > 1e-16:
                result += c * HermitePolynomial.evaluate(x, k)
        return result

def hermite_transform(f: Callable, max_degree: int, use_mpmath: bool = False) -> np.ndarray:
    return HermiteProjection(max_degree, use_mpmath).project(f)

def inverse_hermite_transform(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return HermiteProjection(len(coeffs)-1).approximate(x, coeffs)