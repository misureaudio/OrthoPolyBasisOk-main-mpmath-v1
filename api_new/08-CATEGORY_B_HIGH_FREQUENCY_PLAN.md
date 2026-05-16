# CATEGORY B FIX: High-Frequency Oscillation Undersampling in Non-Laguerre Paths

## Problem Statement

The `_suggest_degree_range()` method classifies **all trigonometric functions as "bounded" derivative growth** (since sin/cos derivatives cycle), which always returns `suggested_max_n = 32` regardless of oscillation frequency. This causes catastrophic undersampling for high-frequency integrands on finite domains.

### Concrete Evidence

| Test Case | Expression | Interval | Cycles | n=32 gives | Needed (~10/half-period) |
|-----------|-----------|----------|--------|------------|--------------------------|
| Leg_Osc_High | sin(40x)+cos(25x) | [-1, 1] | 12.7 | 2.5 nodes/cycle | ~254 |
| Stress_RapidOsc | sin(120x) | [0, 1] | 19.1 | 1.7 nodes/cycle | ~381 |
| Leg_Osc_&#969;80 | cos(80x) | [-1, 1] | 25.5 | 1.3 nodes/cycle | ~509 |

With only 1-2 nodes per cycle, Gauss-Legendre cannot resolve the oscillation and produces garbage values or false convergence (both n=32 and n=64 return wrong but similar values).

## Root Cause Analysis

The problem exists in **two places**:

### Location 1: `_suggest_degree_range()` — no frequency awareness
```python
# Lines 605-629
def _suggest_degree_range(self, analysis: FunctionAnalysis) -> tuple:
    growth = analysis.derivative_growth_rate
    # ...
    if growth == "bounded":
        lo, hi = 8, 32   # <-- ALL trig functions fall here regardless of frequency!
```

### Location 2: `execute_quadrature()` — no oscillation-aware n adjustment
```python
# Lines 677-678
if n is None:
    n = analysis.suggested_max_n  # <-- uses the unadjusted (too-low) value
```

The CATEGORY D fix added `_estimate_max_oscillation_frequency()` and `_compute_oscillation_safe_n()`, but these are only used in the **Laguerre fallback path** and **Hermite fallback path**. The primary Legendre and Chebyshev paths have no oscillation awareness.

## Design: Two-Layer Fix

### Layer 1: Oscillation-aware degree range (analysis phase)
Modify `_suggest_degree_range()` to detect oscillatory integrands and inflate the suggested node count based on frequency content. This ensures `suggested_max_n` reflects actual resolution needs rather than just derivative growth classification.

```python
def _suggest_degree_range(self, analysis: FunctionAnalysis) -> tuple:
    # ... existing logic for lo, hi based on growth rate ...

    # NEW: inflate based on oscillation frequency
    omega_max = self._estimate_max_oscillation_frequency(
        analysis.sympy_expr, analysis.variable,
        analysis.interval_a, analysis.interval_b
    )
    if omega_max > 1e-6:
        interval_width = abs(analysis.interval_b - analysis.interval_b)
        n_half_periods = omega_max * interval_width / math.pi
        min_osc_nodes = int(10 * n_half_periods)
        hi = max(hi, min(min_osc_nodes, 500))  # cap at 500

    return (lo, hi)
```

### Layer 2: Oscillation-aware initial n in execute_quadrature (execution phase)
After computing `n` from analysis, apply oscillation-safe adjustment before calling the quadrature routine. This is a safety net in case someone passes an explicit n that's too low for the frequency content.

```python
def execute_quadrature(self, ...):
    # ... existing setup ...

    if n is None:
        n = analysis.suggested_max_n

    # NEW: ensure n is sufficient for oscillation frequency
    omega_max = self._estimate_max_oscillation_frequency(
        expr, variable, a, b
    )
    if omega_max > 1e-6:
        interval_width = abs(b - a)
        n_half_periods = omega_max * interval_width / math.pi
        min_osc_nodes = int(10 * n_half_periods)
        n = max(n, min(min_osc_nodes, 500))

    # ... rest of execute_quadrature unchanged ...
```

### Why Two Layers?
- **Layer 1** ensures the analysis phase reports correct node requirements (used by API middleware which calls `analyze()` then uses `suggested_max_n`)
- **Layer 2** is a safety net for direct calls where someone might pass an explicit low n

## Implementation Plan

### Step 1: Modify `_suggest_degree_range()` in `quadrature_analyzer_d_adapted.py`
Add oscillation frequency detection after the existing growth-rate logic. Use the already-implemented `_estimate_max_oscillation_frequency()`.

### Step 2: Modify `execute_quadrature()` in `quadrature_analyzer_d_adapted.py`
After determining initial n (from analysis or explicit parameter), apply oscillation-safe adjustment using `_compute_oscillation_safe_n()`.

### Step 3: Run test suite to verify improvement

## Expected Impact on Test Cases

| Test Case | Current Status | After Fix |
|-----------|---------------|-----------|
| Leg_Osc_High (sin(40x)+cos(25x)) | n=32, ~2.5 nodes/cycle | n~254, ~10 nodes/half-period |
| Stress_RapidOsc (sin(120x)) | n=64, ~3.3 nodes/cycle | n~381, ~10 nodes/half-period |
| Leg_Osc_&#969;80 (cos(80x)) | n=32, ~1.3 nodes/cycle | n=500 (capped), ~7.8 nodes/half-period |

Note: cos(80x) on [-1,1] needs 509 nodes but we cap at 500 — this may still not fully converge to tol=1e-12, but it will be dramatically better than the current 32 nodes. This is a fundamental limitation of Gauss-Legendre for extreme frequencies and would benefit from dedicated oscillatory quadrature methods (Filon, Levin) which are out of scope for this fix.

## Files Modified
- `api_new/quadrature_analyzer_d_adapted.py` — add oscillation awareness to `_suggest_degree_range()` and `execute_quadrature()`

## Files NOT Modified (protected)
- `legendre/`, `chebyshev/`, `hermite/`, `laguerre/` — core quadrature modules untouched
- `api_new/main_mod4intv.py` — server middleware untouched