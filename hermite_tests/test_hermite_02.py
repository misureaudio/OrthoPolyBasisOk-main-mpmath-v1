from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from hermite.high_precision import HermiteMPMath

H50 = HermiteMPMath(50, dps=100)
val = H50.evaluate("2.5") # mp.mpf with 100-digit precision
print(val)