
import sys, os
print('sys.path[0]:', sys.path[0])
print('__file__:', __file__)

try:
    from legendre import LegendreQuadrature
    print('Import SUCCESS')
except ImportError as e:
    print(f'Import FAILED: {e}')
