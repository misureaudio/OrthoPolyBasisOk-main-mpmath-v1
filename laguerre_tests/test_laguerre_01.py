from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from laguerre.symbolic import LaguerreSymbolic
L3 = LaguerreSymbolic(3, alpha=0.5)
print(L3.expression) # L_3^(0.5)(x) as SymPy expression
print(L3.evaluate(2)) # Evaluate at x=2