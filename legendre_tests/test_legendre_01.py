from __future__ import annotations

# import math
import sys
# import numpy as np
# import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

import sympy as sp

from legendre.symbolic import LegendreSymbolic, legendre_symbolic_basis

# Single polynomial
P5 = LegendreSymbolic(5)
print(P5.expression)              # SymPy expression for P_5(x)
print(P5.coefficients_ascending)  # [0, -15/8, 0, 35/8, 0, -35/16] ... exact rationals
print(P5.evaluate(sp.Rational(1,2)))  # Exact evaluation at x=1/2

# Derivative and integral
print(P5.derivative())            # dP_5/dx as SymPy expression
print(P5.integral())              # ∫P_5 dx as SymPy expression

# Basis generation
basis = legendre_symbolic_basis(4)  # [P_0, P_1, P_2, P_3, P_4]