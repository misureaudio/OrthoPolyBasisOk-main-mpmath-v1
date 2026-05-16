"""LAYER 2 — High precision (mpmath) arithmetic for Laguerre polynomials."""
from __future__ import annotations
from typing import List, Union

try:
    import mpmath as mp
except ImportError:
    raise ImportError("mpmath is required. Install with: pip install mpmath")

class LaguerreMPMath:
    """Arbitrary-precision generalized Laguerre L_n^(α) via mpmath."""

    def __init__(self, n: int, alpha: float = 0.0, dps: int = 50):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if alpha <= -1:
            raise ValueError("alpha must be > -1")
        self.n = n
        self.alpha = alpha
        self.dps = dps

    def evaluate(self, x: Union[float, str, mp.mpf]):
        """Evaluates L_n^(α)(x) using a local precision context."""
        with mp.workdps(self.dps):
            # mpmath uses laguerre(n, a, x) for the generalized form
            # We pass n, alpha, and x in that order.
            prec_alpha = mp.mpf(self.alpha)
            prec_x = mp.mpf(x)
            return mp.laguerre(self.n, prec_alpha, prec_x)

def laguerre_high_precision_basis(max_n: int, alpha: float = 0.0, dps: int = 50) -> List[LaguerreMPMath]:
    return [LaguerreMPMath(n, alpha, dps=dps) for n in range(max_n + 1)]