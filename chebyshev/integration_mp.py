"""LAYER 4 — High-precision (mpmath) quadrature and projection for Chebyshev polynomials.

Provides arbitrary-precision equivalents of the routines in integration.py:
  - Gauss-Chebyshev quadrature (Types I, II, III, IV)
  - Clenshaw-Curtis quadrature with mpmath weights
  - Chebyshev projection and approximation at user-controlled precision
"""
from __future__ import annotations
from typing import Callable, List, Tuple, Union
import mpmath as mp
import numpy as np

# ── Module-level cache keyed by (n, type, dps) ────────────────────────
_chebyshev_mp_cache: dict = {}

# =====================================================================
#  Gauss-Chebyshev Quadrature (mpmath)
# =====================================================================

class GaussChebyshevQuadrature:
    """Gauss-Chebyshev quadrature with arbitrary-precision nodes/weights.

    Type I  — weight w(x) = 1/sqrt(1-x^2)
    Type II — weight w(x) = sqrt(1-x^2)
    """

    def __init__(self, n: int, quad_type: str = "I", dps: int = 80):
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        self.n = n
        self.quad_type = quad_type.upper()
        self.dps = dps

        cache_key = (n, self.quad_type, dps)
        if cache_key not in _chebyshev_mp_cache:
            _chebyshev_mp_cache[cache_key] = self._compute_nodes_weights()
        self._nodes, self._weights = _chebyshev_mp_cache[cache_key]

    def _compute_nodes_weights(self) -> Tuple[List[mp.mpf], List[mp.mpf]]:
        with mp.workdps(self.dps):
            if self.n == 0:
                # n=0 is the midpoint rule with the integral of the weight function
                # Integral 1/sqrt(1-x^2) = pi | Integral sqrt(1-x^2) = pi/2
                w_sum = mp.pi if self.quad_type == "I" else mp.pi / 2
                return [mp.mpf(0)], [mp.mpf(w_sum)]

            pi = mp.pi
            n = self.n

            if self.quad_type == "I":
                # Nodes: cos(pi*(2k-1)/(2n)), k=1..n
                nodes = [mp.cos(pi * (2 * mp.mpf(k) - 1) / (2 * n)) for k in range(1, n + 1)]
                weights = [pi / n] * n

            elif self.quad_type == "II":
                # Nodes: cos(k*pi/(n+1)), k=1..n
                # Weights: (pi/(n+1)) * sin^2(k*pi/(n+1))
                nodes = [mp.cos(pi * mp.mpf(k) / (n + 1)) for k in range(1, n + 1)]
                weights = [
                    (pi / (n + 1)) * mp.sin(pi * mp.mpf(k) / (n + 1)) ** 2
                    for k in range(1, n + 1)
                ]
            else:
                raise ValueError(f"Unsupported type '{self.quad_type}'")

            # Sort by ascending node value
            combined = sorted(zip(nodes, weights), key=lambda x: x[0])
            return [x for x, _ in combined], [w for _, w in combined]

    def integrate(self, f: Callable[[mp.mpf], mp.mpf]) -> mp.mpf:
        with mp.workdps(self.dps):
            return sum(w * f(x) for x, w in zip(self._nodes, self._weights))

# =====================================================================
#  Clenshaw-Curtis Quadrature (mpmath)
# =====================================================================
class ClenshawCurtisMP:
    """Clenshaw-Curtis quadrature with mpmath-precision weights."""

    def __init__(self, n: int, dps: int = 80):
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        self.n = n
        self.dps = dps

        cache_key = ("cc", n, dps)
        if cache_key not in _chebyshev_mp_cache:
            _chebyshev_mp_cache[cache_key] = self._compute_cc_weights()
        self._nodes, self._weights = _chebyshev_mp_cache[cache_key]

    def _compute_cc_weights(self) -> Tuple[List[mp.mpf], List[mp.mpf]]:
        """Compute CC weights using direct cosine series for high-precision stability."""
        with mp.workdps(self.dps):
            if self.n == 0:
                return [mp.mpf(0)], [mp.mpf(2)]
            
            n = self.n
            nodes = [mp.cos(mp.pi * mp.mpf(k) / n) for k in range(n + 1)]
            weights = []
            
            for k in range(n + 1):
                ck = 1 if (k == 0 or k == n) else 2
                s = mp.mpf(1)
                for j in range(1, (n // 2) + 1):
                    val = mp.mpf(2) / (1 - 4 * mp.mpf(j)**2)
                    bj = 0.5 if (2 * j == n) else 1.0
                    s += bj * val * mp.cos(2 * mp.mpf(j) * k * mp.pi / n)
                weights.append((mp.mpf(ck) / n) * s)
                
            return nodes, weights

    def integrate(self, f: Callable[[mp.mpf], mp.mpf]) -> mp.mpf:
        with mp.workdps(self.dps):
            return sum(w * f(x) for x, w in zip(self._nodes, self._weights))

    def integrate_on_interval(
        self, f: Callable[[mp.mpf], mp.mpf], a: Union[float, str, mp.mpf], b: Union[float, str, mp.mpf]
    ) -> mp.mpf:
        with mp.workdps(self.dps):
            a_mp, b_mp = mp.mpf(a), mp.mpf(b)
            if a_mp == b_mp: return mp.mpf(0)
            jacobian = (b_mp - a_mp) / 2
            midpoint = (b_mp + a_mp) / 2
            return jacobian * sum(w * f(jacobian * x + midpoint) for x, w in zip(self._nodes, self._weights))

# =====================================================================
#  Chebyshev Projection / Approximation (mpmath)
# =====================================================================
class ChebyshevProjectionMP:
    """Project function onto Chebyshev basis using Discrete Cosine Transform (DCT-I)."""

    def __init__(self, max_degree: int, dps: int = 80):
        self.max_degree = max_degree
        self.dps = dps
        # Orthogonality nodes (extrema)
        self._nodes = [mp.cos(mp.pi * mp.mpf(k) / max_degree) for k in range(max_degree + 1)] if max_degree > 0 else [mp.mpf(0)]

    def project(self, f: Callable[[mp.mpf], mp.mpf]) -> List[mp.mpf]:
        """Compute expansion coefficients a_k using DCT-I logic."""
        with mp.workdps(self.dps):
            n = self.max_degree
            if n == 0:
                return [f(mp.mpf(0))]
            
            coeffs = []
            # Calculate values of f at extrema once
            f_vals = [f(x) for x in self._nodes]
            
            for k in range(n + 1):
                s = mp.mpf(0)
                for j in range(n + 1):
                    # Endpoints are weighted 0.5 in DCT-I
                    ej = 0.5 if (j == 0 or j == n) else 1.0
                    s += ej * f_vals[j] * mp.cos(mp.pi * k * j / n)
                
                ek = 0.5 if (k == 0 or k == n) else 1.0
                coeffs.append((2.0 * ek / n) * s)
            return coeffs

    def approximate(self, x: Union[float, str, mp.mpf], coeffs: List[mp.mpf]) -> mp.mpf:
        """Evaluate the series at point x using Clenshaw summation."""
        with mp.workdps(self.dps):
            x_mp = mp.mpf(x)
            n = len(coeffs) - 1
            if n < 0: return mp.mpf(0)

            b_k2, b_k1 = mp.mpf(0), mp.mpf(0)
            for k in range(n, 0, -1):
                b_k = coeffs[k] + 2 * x_mp * b_k1 - b_k2
                b_k2, b_k1 = b_k1, b_k
            return coeffs[0] + x_mp * b_k1 - b_k2

# =====================================================================
#  Convenience Functions (mirroring integration.py API)
# =====================================================================
def chebyshev_transform_mp(f, max_degree, dps=80):
    return ChebyshevProjectionMP(max_degree, dps=dps).project(f)

def inverse_chebyshev_transform_mp(coeffs, x, dps=80):
    return ChebyshevProjectionMP(len(coeffs)-1, dps=dps).approximate(x, coeffs)

def clencurt_mp(f: Callable[[mp.mpf], mp.mpf], n: int, dps: int = 80) -> mp.mpf:
    """Clenshaw-Curtis quadrature on [-1, 1] at arbitrary precision.

    Direct mpmath equivalent of `clencurt_quadrature()` in integration.py.
    """
    return ClenshawCurtisMP(n, dps=dps).integrate(f)

def clencurt_mp_interval(f, a, b, n=32, dps=80):
    return ClenshawCurtisMP(n, dps=dps).integrate_on_interval(f, a, b)

def gauss_chebyshev_mp(f, n, quad_type="I", dps=80):
    return GaussChebyshevQuadrature(n, quad_type=quad_type, dps=dps).integrate(f)

def get_nodes_weights_mp(n, dps=80):
    cc = ClenshawCurtisMP(n, dps=dps)
    return cc._nodes, cc._weights

def map_nodes_to_interval_mp(
    nodes: List[mp.mpf], a: Union[float, str, mp.mpf], b: Union[float, str, mp.mpf], dps: int = 80
) -> List[mp.mpf]:
    """Map Chebyshev nodes from [-1, 1] to [a, b] using mpmath arithmetic."""
    with mp.workdps(dps):
        a_mp = mp.mpf(a)
        b_mp = mp.mpf(b)
        return [(b_mp - a_mp) / 2 * x + (b_mp + a_mp) / 2 for x in nodes]


# =====================================================================
#  Float64 bridge — convert MP results to NumPy when needed
# =====================================================================

def clencurt_quadrature_float(f: Callable[[float], float], n: int, dps: int = 80) -> float:
    """Clenshaw-Curtis quadrature returning a Python float.

    Internally uses mpmath at *dps* precision, then casts the result to float64.
    Useful for drop-in replacement of `clencurt_quadrature()` when higher
    intermediate precision is desired but the final answer is double-precision.
    """
    def f_mp(x):
        return mp.mpf(f(float(x)))
    return float(clencurt_mp(f_mp, n, dps))


def chebyshev_transform_float(
    f: Callable[[float], float], max_degree: int, dps: int = 80
) -> np.ndarray:
    """Chebyshev projection returning a NumPy float64 array.

    Internally uses mpmath at *dps* precision for the quadrature.
    """
    def f_mp(x):
        return mp.mpf(f(float(x)))
    coeffs_mp = chebyshev_transform_mp(f_mp, max_degree, dps)
    return np.array([float(c) for c in coeffs_mp])