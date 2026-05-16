"""Hermite Polynomial Package — 4-Layer Architecture (HP-Z4)."""

from .symbolic import HermiteSymbolic, hermite_symbolic_basis
from .high_precision import HermiteMPMath, hermite_high_precision_basis
from .numerical import HermitePolynomial, hermite_numerical_basis
from .integration import (
    GaussHermiteQuadrature,
    HermiteProjection,
    hermite_transform,
    inverse_hermite_transform,
)

__all__ = [
    "HermiteSymbolic", "hermite_symbolic_basis",
    "HermiteMPMath", "hermite_high_precision_basis",
    "HermitePolynomial", "hermite_numerical_basis",
    "GaussHermiteQuadrature", "HermiteProjection",
    "hermite_transform", "inverse_hermite_transform",
]

__version__ = "2.0.1-hpz4-fixed"