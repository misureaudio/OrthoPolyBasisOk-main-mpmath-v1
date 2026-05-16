"""Laguerre Polynomial Package — 4-Layer Architecture (HP-Z4).

    symbolic.py          LAYER 1  Exact rational coefficients via SymPy
    high_precision.py    LAYER 2  50+ decimal precision via mpmath
    numerical.py         LAYER 3  Fast NumPy array operations
    integration.py       LAYER 4  Quadrature and projection orchestrator

All layers are independent (no cross-imports between L1-L3).
Only integration.py imports from lower layers as the orchestrator.
"""

# ── LAYER 1 — Symbolic ──────────────────────────────────────────────
from .symbolic import (
    LaguerreSymbolic,
    laguerre_symbolic_basis,
)

# ── LAYER 2 — High Precision ────────────────────────────────────────
from .high_precision import (
    LaguerreMPMath,
    laguerre_high_precision_basis,
)

# ── LAYER 3 — Numerical ─────────────────────────────────────────────
from .numerical import (
    LaguerrePolynomial,
    GeneralizedLaguerrePolynomial,
    laguerre_numerical_basis,
)

# ── LAYER 4 — Integration / Quadrature ──────────────────────────────
from .integration import (
    LaguerreQuadrature,
    LaguerreBasis,
    GeneralizedLaguerreBasis,
    compute_roots,
    gauss_quadrature_weights,
    function_projection,
    function_approximation,
)

__all__ = [
    "LaguerreSymbolic",
    "laguerre_symbolic_basis",
    "LaguerreMPMath",
    "laguerre_high_precision_basis",
    "LaguerrePolynomial",
    "GeneralizedLaguerrePolynomial",
    "laguerre_numerical_basis",
    "LaguerreQuadrature",
    "LaguerreBasis",
    "GeneralizedLaguerreBasis",
    "compute_roots",
    "gauss_quadrature_weights",
    "function_projection",
    "function_approximation",
]

__version__ = "2.0.0-hpz4"