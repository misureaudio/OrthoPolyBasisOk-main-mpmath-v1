from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from hermite.symbolic import HermiteSymbolic

H4 = HermiteSymbolic(4)
print(H4.expression) # 16x^4 - 48x^2 + 12
print(H4.evaluate(0)) # 12