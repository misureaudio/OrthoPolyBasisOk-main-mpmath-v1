"""LAYER 2 — High precision (mpmath) arithmetic for Hermite polynomials."""
from __future__ import annotations
from typing import List, Union
import mpmath as mp

class HermiteMPMath:
    """Arbitrary-precision Hermite H_n via mpmath (physicists' convention)."""

    def __init__(self, n: int, dps: int = 50):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self.n = n
        self.dps = dps

    def evaluate(self, x: Union[float, str, mp.mpf]) -> mp.mpf:
        """Evaluate H_n(x) with high precision without leaking global state."""
        with mp.workdps(self.dps):
            # Ensure x is cast to high-precision mpf within this context
            return mp.hermite(self.n, mp.mpf(x))

    def __call__(self, x):
        return self.evaluate(x)

def hermite_high_precision_basis(max_n: int, dps: int = 50) -> List[HermiteMPMath]:
    return [HermiteMPMath(n, dps=dps) for n in range(max_n + 1)]