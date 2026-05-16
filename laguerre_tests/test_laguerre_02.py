from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

# Ensure the package root is on the path
sys.path.insert(0, "..")

from laguerre.high_precision import LaguerreMPMath
# import mpmath as mp

# Create a high-precision Laguerre polynomial L_20^(1.5) with 80 decimal places
L20 = LaguerreMPMath(20, alpha=1.5, dps=80)

# High-precision evaluation at x = 3.7
x_value = "3.7"
val = L20.evaluate(x_value)

# Readable prettyprint - use mp.nstr for full precision display
val_str = mp.nstr(val, n=80)

print("=" * 60)
print("  Laguerre Polynomial Evaluation")
print("=" * 60)
print(f"  Polynomial : L_n^(alpha)(x)")
print(f"  Degree (n) : 20")
print(f"  Parameter  : alpha = 1.5")
print(f"  Point (x)  : {x_value}")
print(f"  Precision  : 80 decimal places")
print("-" * 60)
print(f"  L_20^(1.5)(3.7) = {val_str}")
print("=" * 60)
