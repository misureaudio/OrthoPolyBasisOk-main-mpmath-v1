"""
Chebyshev HP-Z4 Architecture
L1: Symbolic (Exact)
L2: High Precision (mpmath - 50 dps)
L3: Numerical (NumPy - Fast)
L4: Integration (Quadrature)
"""

from .symbolic import ChebyshevSymbolic
from .high_precision import ChebyshevMPMath, get_mpmath_chebyshev_basis
from .numerical import ChebyshevPolynomial, ChebyshevGenerator
from .integration import (
    ChebyshevQuadrature, 
    clencurt_quadrature, 
    clencurt_integrate_interval  # Added for Layer 4 parity
)

__all__ = [
    "ChebyshevSymbolic",
    "ChebyshevMPMath",
    "ChebyshevPolynomial",
    "ChebyshevGenerator",
    "ChebyshevQuadrature",
    "get_mpmath_chebyshev_basis",
    "clencurt_quadrature",
    "clencurt_integrate_interval" # Now accessible as chebyshev.clencurt_integrate_interval
]

__version__ = "2.0.3-hpz4"