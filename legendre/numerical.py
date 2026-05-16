"""LAYER 3 — Fast numerical (NumPy) operations for Legendre polynomials.

Provides three-term recurrence evaluation, coefficient caching, and vectorized
batch processing. No internal dependencies on other layers.
"""
from __future__ import annotations
from typing import List, Union
import warnings
import numpy as np

def _validate_n(n: int) -> None:
    """Validate that n is a non-negative integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

class LegendrePolynomial:
    """Fast Legendre polynomial via three-term recurrence (NumPy)."""

    def __init__(self, n: int):
        _validate_n(n)
        self.n = n
        self._coeffs_desc = None
        self._coeffs_asc = None

    @staticmethod
    def evaluate(x: Union[float, np.ndarray], n: int) -> Union[float, np.ndarray]:
        """Evaluate P_n(x) using forward three-term recurrence. Stable for all n."""
        _validate_n(n)
        if n == 0:
            return np.ones_like(np.atleast_1d(x)).flatten() if isinstance(x, np.ndarray) else 1.0

        is_arr = isinstance(x, np.ndarray)
        x_val = np.asarray(x, dtype=np.float64)

        p_prev = np.ones_like(x_val)   # P_0
        p_curr = x_val.copy()          # P_1

        for k in range(2, n + 1):
            p_next = ((2 * k - 1) * x_val * p_curr - (k - 1) * p_prev) / k
            p_prev, p_curr = p_curr, p_next

        return p_curr if is_arr else float(p_curr.item())

    def __call__(self, x):
        return self.evaluate(x, self.n)

    @property
    def coefficients_descending(self) -> List[float]:
        """Coefficients [c_0, ..., c_n] for P_n(x) = sum c_i * x^{n-i}."""
        if self._coeffs_desc is None:
            self._coeffs_desc = _LEGENDRE_GEN.get_coefficients_descending(self.n)
        return self._coeffs_desc.copy()

    @property
    def coefficients_ascending(self) -> List[float]:
        """Coefficients [c_0, ..., c_n] for P_n(x) = sum c_i * x^i."""
        if self._coeffs_asc is None:
            self._coeffs_asc = _LEGENDRE_GEN.get_coefficients_ascending(self.n)
        return self._coeffs_asc.copy()

    def derivative_coefficients(self) -> List[float]:
        """Coefficients of P_n'(x) in descending order."""
        if self.n == 0:
            return [0.0]
        coeffs = self.coefficients_descending
        degree = len(coeffs) - 1
        deriv = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power > 0:
                deriv.append(c * power)
        return deriv

    def integral_coefficients(self) -> List[float]:
        """Coefficients of ∫P_n(x)dx in descending order (constant = 0)."""
        coeffs = self.coefficients_descending
        degree = len(coeffs) - 1
        integ = []
        for i, c in enumerate(coeffs):
            power = degree - i
            integ.append(c / (power + 1))
        integ.append(0.0)
        return integ

class LegendreGenerator:
    """Cached generator for Legendre polynomial coefficients."""
    def __init__(self):
        self._cache_desc = {}
        self._cache_asc = {}

    def get_coefficients_descending(self, n: int) -> List[float]:
        _validate_n(n)
        if n in self._cache_desc:
            return self._cache_desc[n].copy()
        if n == 0: return [1.0]
        if n == 1: return [1.0, 0.0]

        p_prev2 = [1.0]
        p_prev1 = [1.0, 0.0]
        for k in range(2, n + 1):
            current = [0.0] * (k + 1)
            f1, f2 = (2 * k - 1), (k - 1)
            for i, coeff in enumerate(p_prev1):
                current[i] += (f1 * coeff) / k
            for i, coeff in enumerate(p_prev2):
                current[i+2] -= (f2 * coeff) / k
            p_prev2, p_prev1 = p_prev1, current

        self._cache_desc[n] = p_prev1
        return p_prev1.copy()

    def get_coefficients_ascending(self, n: int) -> List[float]:
        _validate_n(n)
        return self.get_coefficients_descending(n)[::-1]

_LEGENDRE_GEN = LegendreGenerator()

# --- Convenience Functions ---

def legendre_derivative(n: int) -> List[float]:
    """Return coefficients of P_n'(x) in descending order."""
    _validate_n(n)
    return LegendrePolynomial(n).derivative_coefficients()

def legendre_integral(n: int) -> List[float]:
    """Return coefficients of ∫P_n(x)dx in descending order."""
    _validate_n(n)
    return LegendrePolynomial(n).integral_coefficients()

def legendre_polynomial(n: int, x: float) -> float:
    """Evaluate P_n(x)."""
    return LegendrePolynomial.evaluate(x, n)

def legendre_coefficients_descending(n: int) -> List[float]:
    """Get descending monomial coefficients."""
    return _LEGENDRE_GEN.get_coefficients_descending(n)

def legendre_coefficients_ascending(n: int) -> List[float]:
    """Get ascending monomial coefficients."""
    return _LEGENDRE_GEN.get_coefficients_ascending(n)

def legendre_coefficients(n: int) -> List[float]:
    """Get descending monomial coefficients. Alias for descending."""
    return legendre_coefficients_descending(n)

def generate_numpy_legendre(n: int) -> np.ndarray:
    """Get descending coefficients as NumPy array."""
    return np.array(legendre_coefficients_descending(n))

def generate_numpy_legendre_descending(n: int) -> np.ndarray:
    """Get descending coefficients as NumPy array."""
    return np.array(legendre_coefficients_descending(n))

def generate_numpy_legendre_ascending(n: int) -> np.ndarray:
    """Get ascending coefficients as NumPy array."""
    return np.array(legendre_coefficients_ascending(n))

def get_numpy_legendre_basis(max_n: int) -> List[LegendrePolynomial]:
    """Return basis of LegendrePolynomial instances [P_0, ..., P_{max_n}]."""
    return [LegendrePolynomial(n) for n in range(max_n + 1)]

def get_numpy_legendre_basis_descending(max_n: int) -> List[np.ndarray]:
    """Return basis as descending coefficient arrays."""
    return [np.array(legendre_coefficients_descending(n)) for n in range(max_n + 1)]

def get_numpy_legendre_basis_ascending(max_n: int) -> List[np.ndarray]:
    """Return basis as ascending coefficient arrays."""
    return [np.array(legendre_coefficients_ascending(n)) for n in range(max_n + 1)]

def evaluate_numpy_legendre(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Evaluate P_n(x) via NumPy. Convenience wrapper."""
    return LegendrePolynomial.evaluate(x, n)

def legendre_numerical_basis(max_n: int) -> List[LegendrePolynomial]:
    """Alias for get_numpy_legendre_basis."""
    return get_numpy_legendre_basis(max_n)