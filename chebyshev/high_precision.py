"""LAYER 2 — High precision (mpmath) arithmetic for Chebyshev polynomials."""
from __future__ import annotations
from typing import List, Union
import mpmath as mp

class ChebyshevMPMath:
    """Arbitrary-precision Chebyshev T_n via mpmath with context management."""

    def __init__(self, n: int, dps: int = 50):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self.n = n
        self.dps = dps

    def evaluate(self, x: Union[float, str, mp.mpf]) -> mp.mpf:
        """Evaluate T_n(x) at specific precision."""
        with mp.workdps(self.dps):
            return mp.chebyt(self.n, mp.mpf(x))

    def derivative(self, x: Union[float, str, mp.mpf]) -> mp.mpf:
        """Evaluate T'_n(x) using high-precision numerical differentiation."""
        with mp.workdps(self.dps):
            # Identity: T'_n(x) = n * U_{n-1}(x)
            # Or use mpmath's diff if n=0 case is handled
            if self.n == 0:
                return mp.mpf(0)
            return mp.diff(lambda t: mp.chebyt(self.n, t), mp.mpf(x))

    def integral(self, x: Union[float, str, mp.mpf]) -> mp.mpf:
        """Evaluate integral of T_n(x) from 0 to x."""
        with mp.workdps(self.dps):
            return mp.quad(lambda t: mp.chebyt(self.n, t), [0, mp.mpf(x)])

    def get_coefficients(self) -> List[mp.mpf]:
        """Returns exact monomial coefficients as mpmath floats."""
        with mp.workdps(self.dps):
            # We can use the recurrence to generate exact mpf coefficients
            if self.n == 0: return [mp.mpf(1)]
            if self.n == 1: return [mp.mpf(0), mp.mpf(1)]
            
            c_prev2 = [mp.mpf(1)]
            c_prev1 = [mp.mpf(0), mp.mpf(1)]
            for _ in range(2, self.n + 1):
                curr = [mp.mpf(0)] * (len(c_prev1) + 1)
                for i, val in enumerate(c_prev1):
                    curr[i+1] += 2 * val
                for i, val in enumerate(c_prev2):
                    curr[i] -= val
                c_prev2, c_prev1 = c_prev1, curr
            return c_prev1

def get_mpmath_chebyshev_basis(max_n: int, dps: int = 50) -> List[ChebyshevMPMath]:
    return [ChebyshevMPMath(n, dps=dps) for n in range(max_n + 1)]