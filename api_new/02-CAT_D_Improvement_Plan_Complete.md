## CATEGORY D Improvement Plan — Complete

**Document:** `CATEGORY_D_LAGUERRE_WEIGHT_MISMATCH_PLAN.md`

### Key findings from the updated test log (19/33 fidelity after Category E fix):

Of the 4 remaining Laguerre failures, only **one is a genuine value mismatch**:
- **Stress_HighFreqLag** (`sin(50x)·e^(-0.9x)`): Direct=−0.40 vs API=+0.60 — both wrong AND disagreeing

The other 3 are convergence-flag mismatches with correct values (Category A, not D).

### Root cause of Stress_HighFreqLag:
Three compounding issues:
1. Weight mismatch forces fallback to finite-window Legendre on [0, L≈38]
2. `_probe_derivative_growth` returns "bounded" for sin(50x)·e^(-0.9x), recommending only n∈[8,32]
3. No frequency-aware node selection — 64 nodes to resolve ~80 oscillation cycles = catastrophic undersampling

### Proposed fix (new file `quadrature_analyzer_d_adapted.py`):
1. **`_extract_oscillation_frequencies()`** — detect ω from sin/cos atoms
2. **`_compute_oscillation_safe_n()`** — compute min nodes: 10·ω·L/π (capped at 500)
3. Apply to both Laguerre and Hermite fallback paths

### Expected result:
- Stress_HighFreqLag: both paths → ~0.020 (analytical: 50/2500.81 ≈ 0.020)
- Lag_Osc_Decay: both paths → ~0.0049 (analytical: 0.7/144.49 ≈ 0.0049)

### Constraint honored:
No modifications to `quadrature_analyzer_laguerre_fixed.py` or any module in chebyshev/, hermite/, laguerre/, legendre/. All changes go into a new copy file.