"""LAYER 3 — Fast numerical (NumPy) operations for Chebyshev polynomials."""
from __future__ import annotations
from typing import List
import numpy as np

class ChebyshevGenerator:
    """Cached engine for Chebyshev polynomial coefficients."""
    def __init__(self):
        self._cache = {}

    def get_derivative_series(self, n: int) -> List[float]:
        if n == 0: return [0.0]
        deriv_coeffs = [0.0] * n
        for i in range(n - 1, -1, -2):
            deriv_coeffs[i] = float(n) if i == 0 else float(2 * n)
        return deriv_coeffs

    def get_integral_series(self, n: int) -> List[float]:
        integ_coeffs = [0.0] * (n + 2)
        if n == 0: integ_coeffs[1] = 1.0
        elif n == 1:
            integ_coeffs[0], integ_coeffs[2] = 0.25, 0.25
        else:
            integ_coeffs[n + 1] = 1.0 / (2 * (n + 1))
            integ_coeffs[n - 1] = -1.0 / (2 * (n - 1))
        return integ_coeffs

    def evaluate_series(self, x: float, series_coeffs: List[float]) -> float:
        if not series_coeffs: return 0.0
        n = len(series_coeffs) - 1
        b_k, b_k1, b_k2 = 0.0, 0.0, 0.0
        for k in range(n, 0, -1):
            b_k = series_coeffs[k] + 2 * x * b_k1 - b_k2
            b_k2, b_k1 = b_k1, b_k
        return series_coeffs[0] + x * b_k1 - b_k2

    def get_monomial_coefficients(self, n: int) -> List[float]:
        if n in self._cache: return self._cache[n].copy()
        if n == 0: coeffs = [1.0]
        elif n == 1: coeffs = [0.0, 1.0]
        else:
            p2, p1 = [1.0], [0.0, 1.0]
            for k in range(2, n + 1):
                curr = [0.0] * (k + 1)
                for i, c in enumerate(p1): curr[i+1] += 2.0 * c
                for i, c in enumerate(p2): curr[i] -= c
                p2, p1 = p1, curr
            coeffs = p1
        self._cache[n] = coeffs
        return coeffs.copy()

_GEN = ChebyshevGenerator()

class ChebyshevPolynomial:
    """Object-Oriented API for numerical Chebyshev operations."""
    def __init__(self, n: int):
        self.n = n
        self._d_series = _GEN.get_derivative_series(n)
        self._i_series = _GEN.get_integral_series(n)

    def __call__(self, x: float) -> float:
        if self.n == 0: return 1.0
        t0, t1 = 1.0, float(x)
        for _ in range(2, self.n + 1):
            t0, t1 = t1, 2 * x * t1 - t0
        return t1

    def derivative(self, x: float) -> float:
        return _GEN.evaluate_series(x, self._d_series)

    def integral(self, x: float) -> float:
        return _GEN.evaluate_series(x, self._i_series)

    @property
    def coefficients(self) -> np.ndarray:
        return np.array(_GEN.get_monomial_coefficients(self.n))

# ── Standalone Functions (Required by test_chebyshev_03) ─────────────

def chebyshev_derivative_stable(n: int, x: float) -> float:
    return _GEN.evaluate_series(x, _GEN.get_derivative_series(n))

def chebyshev_integral_stable(n: int, x: float) -> float:
    return _GEN.evaluate_series(x, _GEN.get_integral_series(n))

def chebyshev_coefficients(n: int) -> List[float]:
    return _GEN.get_monomial_coefficients(n)

def generate_numpy_chebyshev(n: int) -> np.ndarray:
    """Returns NumPy array of monomial coefficients."""
    return np.array(_GEN.get_monomial_coefficients(n))

def get_numpy_chebyshev_basis(max_n: int) -> List[np.ndarray]:
    """Returns list of NumPy coefficient arrays up to max_n."""
    return [generate_numpy_chebyshev(n) for n in range(max_n + 1)]