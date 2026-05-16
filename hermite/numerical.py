"""LAYER 3 — Fast numerical (NumPy) operations for Hermite polynomials."""
from __future__ import annotations
from typing import Union, List, Optional
import numpy as np

class HermitePolynomial:
    """Fast Hermite polynomial H_n(x) via three-term recurrence (physicists' convention)."""

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self.n = n
        self._coeffs: Optional[List[float]] = None

    @staticmethod
    def evaluate(x: Union[float, np.ndarray], n: int) -> Union[float, np.ndarray]:
        """Evaluate H_n(x) using three-term recurrence: H_{n+1} = 2x*H_n - 2n*H_{n-1}"""
        if n == 0:
            return np.ones_like(np.atleast_1d(x)).flatten() if isinstance(x, np.ndarray) else 1.0

        is_arr = isinstance(x, np.ndarray)
        x_val = np.asarray(x, dtype=np.float64)

        h_prev = np.ones_like(x_val)    # H_0
        h_curr = 2 * x_val              # H_1

        if n == 1:
            return h_curr if is_arr else float(h_curr.item())

        for k in range(2, n + 1):
            h_next = 2 * x_val * h_curr - 2 * (k - 1) * h_prev
            h_prev, h_curr = h_curr, h_next

        return h_curr if is_arr else float(h_curr.item())

    def __call__(self, x):
        return self.evaluate(x, self.n)

    @property
    def coefficients_ascending(self) -> List[float]:
        """Monomial coefficients [c_0, c_1, ..., c_n] for H_n(x)."""
        if self._coeffs is None:
            if self.n == 0:
                self._coeffs = [1.0]
            elif self.n == 1:
                self._coeffs = [0.0, 2.0]
            else:
                c_prev = [1.0]   # H_0
                c_curr = [0.0, 2.0]  # H_1
                for k in range(2, self.n + 1):
                    c_next = [0.0] * (k + 1)
                    for i, coeff in enumerate(c_curr):
                        c_next[i + 1] += 2.0 * coeff
                    for i, coeff in enumerate(c_prev):
                        c_next[i] -= 2.0 * (k - 1) * coeff
                    c_prev = c_curr
                    c_curr = c_next
                self._coeffs = c_curr
        return self._coeffs.copy()

# THIS WAS THE MISSING PART CAUSING THE IMPORT ERROR
def hermite_numerical_basis(max_n: int) -> List[HermitePolynomial]:
    """Return [H_0, H_1, ..., H_{max_n}] as HermitePolynomial instances."""
    return [HermitePolynomial(n) for n in range(max_n + 1)]