"""Legendre Polynomial Package — 4-Layer Architecture (HP-Z4).

    symbolic.py          LAYER 1  Exact rational coefficients via SymPy
    high_precision.py    LAYER 2  50+ decimal precision via mpmath
    numerical.py         LAYER 3  Fast NumPy/SciPy array operations
    integration.py       LAYER 4  Quadrature and projection orchestrator

All layers are independent (no cross-imports between L1-L3).
Only integration.py imports from lower layers as the orchestrator.
"""

# ── LAYER 1 — Symbolic ──────────────────────────────────────────────
from .symbolic import (
    LegendreSymbolic,
    legendre_symbolic_basis,
    generate_sympy_legendre,
    get_sympy_legendre_basis,
)

# ── LAYER 2 — High Precision ────────────────────────────────────────
from .high_precision import (
    LegendreMPMath,
    legendre_high_precision_basis,
    generate_mpmath_legendre,
    get_mpmath_legendre_basis,
    get_mpmath_legendre_coefficients,
    get_mpmath_legendre_coefficients_descending,
    get_mpmath_legendre_coefficients_ascending,
    evaluate_mpmath_legendre,
)

# ── LAYER 3 — Numerical ─────────────────────────────────────────────
from .numerical import (
    LegendrePolynomial,
    LegendreGenerator,
    legendre_polynomial,
    legendre_coefficients,
    legendre_coefficients_descending,
    legendre_coefficients_ascending,
    legendre_derivative,
    legendre_integral,
    generate_numpy_legendre,
    generate_numpy_legendre_descending,
    generate_numpy_legendre_ascending,
    get_numpy_legendre_basis,
    get_numpy_legendre_basis_descending,
    get_numpy_legendre_basis_ascending,
    evaluate_numpy_legendre,
)

# ── LAYER 4 — Integration / Quadrature ──────────────────────────────
from .integration import (
    GaussLegendreQuadrature,
    LegendreQuadrature,
    HighPrecisionGaussLegendre,
    gauss_legendre,
    gauss_legendre_newton,
    gauss_legendre_golub_welsch,
    gauss_legendre_high_precision,
)

__all__ = [
    # Layer 1 — Symbolic
    "LegendreSymbolic",
    "legendre_symbolic_basis",
    "generate_sympy_legendre",
    "get_sympy_legendre_basis",
    # Layer 2 — High Precision
    "LegendreMPMath",
    "legendre_high_precision_basis",
    "generate_mpmath_legendre",
    "get_mpmath_legendre_basis",
    "evaluate_mpmath_legendre",
    "get_mpmath_legendre_coefficients",
    "get_mpmath_legendre_coefficients_descending",
    "get_mpmath_legendre_coefficients_ascending",
    # Layer 3 — Numerical
    "LegendrePolynomial",
    "LegendreGenerator",
    "legendre_polynomial",
    "legendre_coefficients",
    "legendre_coefficients_descending",
    "legendre_coefficients_ascending",
    "legendre_derivative",
    "legendre_integral",
    "generate_numpy_legendre",
    "generate_numpy_legendre_descending",
    "generate_numpy_legendre_ascending",
    "get_numpy_legendre_basis",
    "get_numpy_legendre_basis_descending",
    "get_numpy_legendre_basis_ascending",
    "evaluate_numpy_legendre",
    # Layer 4 — Quadrature
    "GaussLegendreQuadrature",
    "LegendreQuadrature",
    "HighPrecisionGaussLegendre",
    "gauss_legendre",
    "gauss_legendre_newton",
    "gauss_legendre_golub_welsch",
    "gauss_legendre_high_precision",
]

__version__ = "2.0.0-hpz4"