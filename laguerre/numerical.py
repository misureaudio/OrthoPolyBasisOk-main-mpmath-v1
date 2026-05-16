"""LAYER 3 — Fast numerical (NumPy) operations for Laguerre polynomials."""
from __future__ import annotations
import math
import numpy as np
from typing import List

class GeneralizedLaguerrePolynomial:
    def __init__(self, n: int, alpha: float = 0.0):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if alpha <= -1:
            raise ValueError("alpha must be > -1")
        self.n = n
        self.alpha = alpha
        self._coeffs = None

    @staticmethod
    def evaluate_static(x, n: int, alpha: float = 0.0):
        """Standard three-term recurrence."""
        is_arr = isinstance(x, np.ndarray)
        x_val = np.asarray(x, dtype=np.float64) if is_arr else float(x)
        if n == 0: return np.ones_like(x_val) if is_arr else 1.0
        if n == 1: return (alpha + 1 - x_val)
        
        l_p, l_c = 1.0, (alpha + 1 - x_val)
        for k in range(1, n):
            l_n = ((2 * k + alpha + 1 - x_val) * l_c - (k + alpha) * l_p) / (k + 1)
            l_p, l_c = l_c, l_n
        return l_c

    def evaluate(self, x):
        return self.evaluate_static(x, self.n, self.alpha)

    def __call__(self, x):
        return self.evaluate(x)

    @property
    def coefficients_ascending(self) -> List[float]:
        if self._coeffs is None:
            self._coeffs = [
                math.exp(math.lgamma(self.n + self.alpha + 1) - math.lgamma(k + self.alpha + 1) - 
                         math.lgamma(self.n - k + 1) - math.lgamma(k + 1)) * ((-1)**k)
                for k in range(self.n + 1)
            ]
        return self._coeffs.copy()

class LaguerrePolynomial(GeneralizedLaguerrePolynomial):
    def __init__(self, n: int):
        super().__init__(n, alpha=0.0)

def laguerre_numerical_basis(max_n: int, alpha: float = 0.0) -> List[GeneralizedLaguerrePolynomial]:
    return [GeneralizedLaguerrePolynomial(n, alpha) for n in range(max_n + 1)]