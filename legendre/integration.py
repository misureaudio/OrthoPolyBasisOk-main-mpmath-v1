"""LAYER 4 — Quadrature orchestrator for Legendre polynomials.

Auto-selects NumPy vs mpmath based on n. Fixes precision loss in L2.
Adds a lambdify mechanism for high-precision integration from symbolic expressions.
"""
from __future__ import annotations
from typing import List, Tuple, Union, Optional, Any, Callable
import numpy as np
import warnings
import math

try:
    import sympy as sp
    from sympy.abc import x as sympy_x # A default symbol 'x' for lambdify
    from sympy.utilities.lambdify import lambdify
    from sympy import Symbol
except ImportError:
    sp = None
    lambdify = None
    sympy_x = None
    warnings.warn(
        "SymPy is not installed. Symbolic lambdification for high-precision "
        "integration will not be available. Install with: pip install sympy",
        ImportWarning
    )

_gauss_legendre_cache = {}

def _validate_n(n: int) -> None:
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n}")

class GaussLegendreQuadrature:
    """Standard double-precision (NumPy) Gauss-Legendre computation."""

    @staticmethod
    def golub_welsch(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute nodes and weights via Golub-Welsch algorithm.

        Constructs the Jacobi matrix J with beta_k = k/sqrt(4k^2 - 1) on
        off-diagonals (monic Legendre recurrence), then computes:
          - eigenvalues  -> quadrature nodes x_i in (-1, 1)
          - weights      -> 2 * v[0,i]^2  (integral of weight=1 over [-1,1] = 2)
        """
        _validate_n(n)
        if n == 1:
            return np.array([0.0]), np.array([2.0])

        k = np.arange(1, n)
        beta = k / np.sqrt(4.0 * k * k - 1.0)
        J = np.diag(beta, k=1) + np.diag(beta, k=-1)
        nodes, eigenvectors = np.linalg.eigh(J)
        weights = 2.0 * (eigenvectors[0, :] ** 2)
        return nodes, weights

    @staticmethod
    def newton_raphson(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute nodes via Newton-Raphson on P_n(x), then weights from derivative.

        Initial guesses: theta_k = pi*(4k+3)/(4n+2),  k=0..n/2-1  (Trefethen).
        Weight formula:   w_i = 2 / ((1 - x_i^2) * P_n'(x_i)^2).
        Derivative identity: P_n'(x) = n*(x*P_n(x) - P_{n-1}(x)) / (x^2 - 1).
        """
        _validate_n(n)

        # Compute positive roots only, then mirror
        half = (n + 1) // 2
        pos_roots = []
        for k in range(half):
            # Trefethen initial guess: cos(pi*(4k+3)/(4n+2))
            theta = math.pi * (4 * k + 3) / (4 * n + 2)
            x = math.cos(theta)

            # Newton-Raphson iteration
            for _ in range(200):
                p_nm1, p_n = _eval_legendre_pair(x, n)
                # Safeguard against division by zero near +/-1
                if abs(x**2 - 1.0) < 1e-15: # Increased tolerance for stability
                    break
                dp_n = n * (x * p_n - p_nm1) / (x**2 - 1.0)
                if abs(dp_n) < 1e-15: # Increased tolerance for stability
                    break
                dx = p_n / dp_n
                x -= dx
                if abs(dx) < 1e-15:
                    break

            pos_roots.append(x)

        # Build full node list from positive roots (sorted descending)
        # pos_roots[0] is largest, pos_roots[-1] is smallest positive
        nodes = np.zeros(n)
        for i, r in enumerate(pos_roots):
            nodes[i] = -r              # negative half (ascending: most neg first)
            nodes[n - 1 - i] = r       # positive half (ascending: smallest pos last)

        # If n is odd, center root at index n//2 is already 0 from init;
        # but Trefethen only gives positive roots. The zero root for odd n
        # should come from the middle initial guess converging to ~0.
        if n % 2 == 1:
            nodes[n // 2] = 0.0 # Ensure exact zero for odd n

        # Sort ascending (should already be sorted, but ensure)
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]

        # Weights (computed after sorting)
        weights = np.zeros(n)
        for i, xi in enumerate(nodes):
            p_nm1, p_n = _eval_legendre_pair(xi, n)
            if abs(1.0 - xi**2) < 1e-15:
                # Handle endpoints x = +/-1 where x^2-1 is zero.
                # Use limit P_n'(+/-1) = +/- n(n+1)/2
                dp_n = n * (n + 1) / 2.0
                if xi < 0 and (n % 2 == 1): # P_n'(-1) is n(n+1)/2 if n is odd
                    dp_n *= -1
                elif xi < 0 and (n % 2 == 0): # P_n'(-1) is -n(n+1)/2 if n is even
                    dp_n *= -1
            else:
                dp_n = n * (xi * p_n - p_nm1) / (xi**2 - 1.0)
            weights[i] = 2.0 / ((1.0 - xi**2) * dp_n**2)

        return nodes, weights


def _eval_legendre_pair(x: float, n: int) -> Tuple[float, float]:
    """Evaluate P_{n-1}(x) and P_n(x) simultaneously via 3-term recurrence."""
    if n == 0:
        return 0.0, 1.0 # P_{-1} is undefined, but P_0=1
    p_prev = 1.0   # P_0
    p_curr = x     # P_1
    if n == 1:
        return p_prev, p_curr
    for k in range(2, n + 1):
        p_next = ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
        p_prev, p_curr = p_curr, p_next
    return p_prev, p_curr


class LegendreQuadrature:
    """Gauss-Legendre quadrature with automatic precision selection.

    This class provides nodes and weights for integration. When `use_mpmath=True`,
    the nodes and weights are computed with `mpmath` for numerical stability,
    but the results are then cast back to `float64` NumPy arrays.

    The `integrate` method expects a standard Python or NumPy callable
    that accepts `float` or `np.ndarray` inputs. For true arbitrary-precision
    integration, where both nodes, weights, and the integrand's evaluation
    are high-precision `mpmath.mpf` objects, use `HighPrecisionGaussLegendre`.
    """

    def __init__(self, n: int, use_mpmath: bool = False, dps: int = 80):
        self.n = n
        self._use_mpmath = use_mpmath
        self.dps = dps

        cache_key = (n, use_mpmath, dps)
        if cache_key not in _gauss_legendre_cache:
            if use_mpmath:
                _gauss_legendre_cache[cache_key] = self._compute_high_precision_to_float(n, dps)
            else:
                _gauss_legendre_cache[cache_key] = GaussLegendreQuadrature.golub_welsch(n)

        self._nodes, self._weights = _gauss_legendre_cache[cache_key]

    def _compute_high_precision_to_float(self, n: int, dps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute nodes/weights using mpmath for stability, then return as float64 NumPy arrays."""
        import mpmath as mp
        with mp.workdps(int(dps)): # Use requested dps for computation
            nodes_mp, weights_mp = _golub_welsch_mp(n)
            return np.array([float(v) for v in nodes_mp]), np.array([float(v) for v in weights_mp])

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def integrate(self, f: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]) -> float:
        """Perform integration using float64 nodes/weights.

        The callable `f` must accept `float` or `np.ndarray` inputs and return
        `float` or `np.ndarray` outputs, compatible with NumPy operations.
        """
        # For simplicity, this class assumes f is vectorized or can be applied element-wise by numpy
        return float(np.sum(self._weights * f(self._nodes)))

    def integrate_mp(self, f_or_expr, a=-1.0, b=1.0):
        """Full mpmath pipeline for Gauss-Legendre on arbitrary interval [a, b].

        Bridges to HighPrecisionGaussLegendre (which keeps nodes/weights as mp.mpf),
        applies [a,b] interval transformation in mpmath arithmetic, and returns
        raw mp.mpf with NO float() cast.

        Args:
            f_or_expr: A sympy expression or an mpmath-compatible callable.
            a: Left endpoint (default -1).
            b: Right endpoint (default 1).
        Returns:
            mp.mpf — arbitrary-precision result.
        """
        import mpmath as mp

        target_dps = self.dps if hasattr(self, 'dps') else 80
        hp = HighPrecisionGaussLegendre(self.n, dps=target_dps)

        # Get the canonical integrand callable (on [-1,1])
        with mp.workdps(target_dps):
            a_mp = mp.mpf(a)
            b_mp = mp.mpf(b)
            scale = (b_mp - a_mp) / 2
            shift = (b_mp + a_mp) / 2

            # Build mpmath-compatible callable for the canonical interval
            if hasattr(f_or_expr, 'atoms'):
                # sympy expression — lambdify with mpmath modules
                v = Symbol(str(list(f_or_expr.free_symbols)[0])) if f_or_expr.free_symbols else Symbol('x')
                f_mp = lambdify(v, f_or_expr, modules='mpmath')
            elif callable(f_or_expr):
                f_mp = f_or_expr
            else:
                raise TypeError("f_or_expr must be a sympy expression or an mpmath-compatible callable")

            # Interval-transformed integrand: f(scale*t + shift)
            def f_transformed(t):
                return f_mp(scale * t + shift)

            # Integrate on [-1,1] then multiply by Jacobian (scale)
            result = scale * sum(
                w * f_transformed(x) for x, w in zip(hp._nodes, hp._weights)
            )
            return result


def _golub_welsch_mp(n: int) -> Tuple[List[Any], List[Any]]:
    """Golub-Welsch algorithm in pure mpmath for arbitrary precision.

    Uses mp.eigh (symmetric eigensolver) which returns sorted eigenvalues and
    properly structured eigenvector matrix where columns are eigenvectors.
    """
    import mpmath as mp
    if n == 1:
        return [mp.mpf(0)], [mp.mpf(2)]

    # Build Jacobi matrix: diagonal=0, off-diagonal = k/sqrt(4k^2-1)
    J = mp.matrix(n, n)
    for k in range(1, n):
        beta = mp.mpf(k) / mp.sqrt(mp.mpf(4 * k * k - 1))
        J[k - 1, k] = beta
        J[k, k - 1] = beta

    # Use eigh for symmetric matrix — returns (sorted eigenvalues, eigenvector matrix)
    eigenvals, eigenvecs = mp.eigh(J)

    # eigenvals already sorted ascending; columns of eigenvecs are eigenvectors
    nodes = list(eigenvals)
    weights = [mp.mpf(2) * (eigenvecs[0, i] ** 2) for i in range(n)]
    return nodes, weights


class HighPrecisionGaussLegendre:
    """Explicit high-precision Gauss-Legendre quadrature.

    Nodes and weights remain as `mpmath.mpf` objects for true arbitrary-precision
    integration. The `integrate` method can accept either:
    1. A `Callable` that is designed to accept `mpmath.mpf` inputs (e.g., `mp.sin`, `mp.exp`).
    2. A `sympy.Expr` (a symbolic expression), which will be automatically
       lambdified into an `mpmath`-compatible function.
    """
    def __init__(self, n: int, dps: int = 80):
        import mpmath as mp
        self.n = n
        self.dps = dps
        with mp.workdps(dps):
            self._nodes, self._weights = _golub_welsch_mp(n)

    @property
    def nodes(self) -> List[Any]: # List[mp.mpf] but Any to avoid mpmath type hint if not imported
        return self._nodes[:]

    @property
    def weights(self) -> List[Any]: # List[mp.mpf]
        return self._weights[:]

    def integrate(self, f_or_expr: Union[Callable, Any]): # Any for sp.Expr
        """High-precision summation. Integrand must accept mpf objects.

        Args:
            f_or_expr: A callable (e.g., `mp.sin`) that accepts `mpmath.mpf` inputs,
                       or a `sympy.Expr` (e.g., `sp.sin(sympy_x)`).

        Returns:
            The integral value as a Python `float` (after high-precision summation).
        """
        import mpmath as mp
        with mp.workdps(self.dps):
            f_callable = f_or_expr
            if sp and isinstance(f_or_expr, sp.Expr):
                if lambdify is None or sympy_x is None:
                    raise ImportError("SymPy and its lambdify utility are required to lambdify symbolic expressions.")
                f_callable = lambdify(sympy_x, f_or_expr, modules='mpmath')
            elif not callable(f_callable):
                 raise TypeError("Integrand must be a callable or a sympy expression.")

            return sum(w * f_callable(x) for x, w in zip(self._nodes, self._weights))


# --- Convenience Functions ---

def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Legendre nodes and weights (double precision).

    Alias for GaussLegendreQuadrature.golub_welsch.
    """
    return GaussLegendreQuadrature.golub_welsch(n)


def gauss_legendre_newton(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Legendre nodes and weights via Newton-Raphson.

    Uses Newton-Raphson root finding on P_n with Trefethen initial guesses.
    """
    return GaussLegendreQuadrature.newton_raphson(n)


def gauss_legendre_golub_welsch(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Legendre nodes and weights via Golub-Welsch.

    Explicit alias for the Golub-Welsch eigendecomposition method.
    """
    return GaussLegendreQuadrature.golub_welsch(n)


def gauss_legendre_high_precision(n: int, dps: int = 80) -> HighPrecisionGaussLegendre:
    """Factory for high-precision Gauss-Legendre quadrature object."""
    return HighPrecisionGaussLegendre(n, dps=dps)