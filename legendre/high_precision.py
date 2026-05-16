"""LAYER 2 — High precision (mpmath) arithmetic for Legendre polynomials.

Provides configurable decimal precision (50+ digits) via mpmath.
No internal dependencies on other layers.
"""
from __future__ import annotations
from typing import List, Union

try:
    import mpmath as mp
except ImportError:
    raise ImportError("mpmath is required for high_precision layer. Install with: pip install mpmath")


class LegendreMPMath:
    """Arbitrary-precision Legendre polynomial via mpmath."""

    def __init__(self, n: int, dps: int = 50):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self.n = n
        self.dps = dps

    def evaluate(self, x) -> mp.mpf:
        """Evaluate P_n(x) at arbitrary precision."""
        with mp.workdps(self.dps):
            return mp.legendre(self.n, mp.mpf(x))

    def derivative_value(self, x) -> mp.mpf:
        """Numerical derivative P_n'(x) via mpmath.diff with high-precision step size."""
        with mp.workdps(self.dps):
            f = lambda t: mp.legendre(self.n, t)
            return mp.diff(f, mp.mpf(x))

    @property
    def coefficients_ascending(self) -> List[mp.mpf]:
        """Extract monomial coefficients [c_0, c_1...c_n] via high-precision recurrence."""
        with mp.workdps(self.dps):
            if self.n == 0:
                return [mp.mpf(1)]
            if self.n == 1:
                return [mp.mpf(0), mp.mpf(1)]

            # Build ascending coefficients via three-term recurrence
            # P_k = ((2k-1) * x * P_{k-1} - (k-1) * P_{k-2}) / k
            p_prev2 = [mp.mpf(1)]  # P_0
            p_prev1 = [mp.mpf(0), mp.mpf(1)]  # P_1

            for k in range(2, self.n + 1):
                current = [mp.mpf(0)] * (k + 1)
                f1 = mp.mpf(2 * k - 1)
                f2 = mp.mpf(k - 1)
                div_k = mp.mpf(k)

                # Multiply P_{k-1} by (2k-1)x
                for i, c in enumerate(p_prev1):
                    current[i + 1] = (f1 * c) / div_k

                # Subtract (k-1) * P_{k-2}
                for i, c in enumerate(p_prev2):
                    current[i] -= (f2 * c) / div_k

                p_prev2 = p_prev1
                p_prev1 = current

            return p_prev1


# --- Convenience functions ---

def legendre_high_precision_basis(max_n: int, dps: int = 50) -> List[LegendreMPMath]:
    """Return [P_0, ..., P_{max_n}] as LegendreMPMath instances."""
    return [LegendreMPMath(n, dps=dps) for n in range(max_n + 1)]


def generate_mpmath_legendre(n: int, dps: int = 50) -> LegendreMPMath:
    """Return a single high-precision P_n object."""
    return LegendreMPMath(n, dps=dps)


def get_mpmath_legendre_basis(max_n: int, dps: int = 50) -> List[LegendreMPMath]:
    """Return [P_0, ..., P_{max_n}] as LegendreMPMath instances. Alias."""
    return legendre_high_precision_basis(max_n, dps=dps)


def get_mpmath_legendre_coefficients_ascending(n: int, dps: int = 50) -> List[mp.mpf]:
    """Get coefficients in ascending order at high precision."""
    return LegendreMPMath(n, dps=dps).coefficients_ascending


def get_mpmath_legendre_coefficients_descending(n: int, dps: int = 50) -> List[mp.mpf]:
    """Get coefficients in descending order at high precision."""
    return list(reversed(LegendreMPMath(n, dps=dps).coefficients_ascending))


def get_mpmath_legendre_coefficients(n: int, dps: int = 50) -> List[mp.mpf]:
    """Get coefficients (ascending order) at high precision. Alias for ascending."""
    return get_mpmath_legendre_coefficients_ascending(n, dps=dps)


def evaluate_mpmath_legendre(n: int, x, dps: int = 50) -> mp.mpf:
    """Evaluate P_n(x) at arbitrary precision."""
    return LegendreMPMath(n, dps=dps).evaluate(x)