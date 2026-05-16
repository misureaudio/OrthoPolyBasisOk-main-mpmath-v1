Category B (High-Frequency Oscillation Undersampling) fix completed successfully.

**Root Cause**: `_suggest_degree_range()` classified all trigonometric functions as "bounded" derivative growth (since sin/cos derivatives cycle), always returning `suggested_max_n=32` regardless of oscillation frequency. This caused catastrophic undersampling — e.g., cos(80x) on [-1,1] had 25.5 cycles but only 32 nodes (~1.3 per cycle).

**Fix Applied (Two Layers)**:
- **Layer 1**: `_suggest_degree_range()` now detects oscillation frequency via phase derivative method and inflates `suggested_max_n` to ensure ~10 nodes per half-period (capped at 500)
- **Layer 2**: `execute_quadrature()` safety net bumps n upward if explicit n is too low for detected frequency content
- **n2 cap fix**: Changed `min(n*2, 200)` to `min(n*2, 1000)` so convergence checking works correctly when oscillation-aware bump raises n above 100

**Node Count Improvements**:
| Test | Before (n) | After (n) | Cycles | Nodes/Cycle |
|------|-----------|----------|--------|-------------|
| Leg_Osc_High | 32 | 254 | 12.7 | ~8.0 |
| Stress_RapidOsc | 64 | 381 | 19.1 | ~5.0 |
| Leg_Osc_ω80 | 32 | 500 (capped) | 25.5 | ~4.0 |

**Result**: Fidelity score maintained at **33/33 tests passed**. Notably, Leg_Osc_ω80 now converges (Y/Y) with n=500/n2=1000 instead of producing garbage values at n=32.