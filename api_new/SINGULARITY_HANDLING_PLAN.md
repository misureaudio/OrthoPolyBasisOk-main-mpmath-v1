# Singularity Handling Improvement Plan

**Date:** 2026-05-17
**Status:** Proposed — no code modified
**Related Issues:** Phase 5 `Analyze_Singularity` test returns HTTP 400 for `(1-x)**(-0.3)` on `[-1, 1]`

---

## Problem Statement

The expression `(1 - x)**(-0.3)` has an algebraic endpoint singularity at x=1. When the backend's `analyze()` method probes near this point (via `_probe_derivative_growth()` or `_extract_endpoint_singularity_exponents()`), float64 evaluation overflows, raising an uncaught exception that propagates as HTTP 400 from `/analyze`.

**Current workaround:** The test suite avoids the expression by using `(1-x**2)**(-0.3)` on `[-0.9, 0.9]` instead. This is a band-aid — the server should handle genuine endpoint singularities gracefully.

---

## Proposed Changes

### M1: Guard Individual Sub-Analyses in `analyze()`

**File:** `quadrature_analyzer_d_adapted_v2_G3_mpmath.py`, lines ~124–157
**Risk:** Low — only affects error paths; normal behavior unchanged

Wrap each sub-analysis call with try/except so a failure in one does not abort the entire analysis:

```python
# Line 124: singularity detection (already guarded, but verify)
try:
    analysis.singularities = self._find_singularities(expr, a, b)
except Exception:
    analysis.singularities = []

# Line 131: derivative growth probe — NOT currently guarded
try:
    analysis.derivative_growth_rate = self._probe_derivative_growth(expr, a, b, max_deriv_order)
except Exception:
    analysis.derivative_growth_rate = "unknown"

# Lines 152-157: endpoint singularity exponent extraction — NOT currently guarded
if not np.isinf(a) and not np.isinf(b):
    try:
        exp_data = self._extract_endpoint_singularity_exponents(expr, a, b)
        analysis.left_singularity_alpha = exp_data.get("left", 0.0)
        analysis.right_singularity_beta = exp_data.get("right", 0.0)
        analysis.has_algebraic_endpoint_singularity = (
            analysis.left_singularity_alpha > 0 or analysis.right_singularity_beta > 0
        )
    except Exception:
        # Leave defaults: alpha=0, beta=0, has_algebraic_endpoint_singularity=False
        pass
```

**Estimated effort:** ~12 lines added (4 try/except blocks)

---

### M2: Use mpmath for Probe Evaluations in `_extract_endpoint_singularity_exponents()`

**File:** `quadrature_analyzer_d_adapted_v2_G3_mpmath.py`, method starting at line ~1085
**Risk:** Low — uses existing mpmath infrastructure already imported at module level (A1)

The log-log slope method evaluates `f(x)` at points extremely close to endpoints. For `(1-x)**(-0.3)`, evaluating at x=0.999... produces overflow in float64. Replace direct evaluation with an mpmath-aware safe_eval:

```python
def _extract_endpoint_singularity_exponents(self, expr, a: float, b: float) -> dict:
    v = self._var
    # Use mpmath for numerical stability near singularities
    mp.mp.dps = 50
    f_mp = lambdify(v, expr, modules='mpmath')

    def safe_eval(x_val):
        """Evaluate |f(x)| safely; return None on overflow."""
        try:
            val = abs(f_mp(mp.mpf(str(x_val))))
            if not mp.isfinite(val) or val == mp.inf:
                return None
            return float(val)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None

    # For each endpoint, use geometrically spaced probe points
    for endpoint, side in [(a, 'left'), (b, 'right')]:
        alphas = []
        for i in range(1, 20):
            if side == 'left':
                x_val = endpoint + (b - endpoint) * 10**(-i)
            else:
                x_val = endpoint - (endpoint - a) * 10**(-i)

            fval = safe_eval(x_val)
            if fval is not None and fval > 0:
                dist = abs(x_val - endpoint)
                alphas.append((math.log10(dist), math.log10(fval)))

        # Linear regression on log-log data to estimate exponent
        ...  # existing regression logic, but using safe_eval instead of direct eval
```

**Estimated effort:** ~30 lines modified in one method

---

### M3: Server-Side `/analyze` Endpoint Defensive Layer

**File:** `api_new/main_mod4intv_mpmath.py`, `/analyze` endpoint (lines 190–224)
**Risk:** Medium — changes error semantics from HTTP 400 to HTTP 200 with low confidence

Add a secondary try/except that catches analysis failures and returns partial results rather than HTTP 400:

```python
@app.post("/analyze")
async def analyze_function(req: AnalysisRequest):
    try:
        analysis = analyzer.analyze(
            req.expression, interval=req.interval,
            variable=req.variable, max_deriv_order=req.max_deriv_order
        )
        resp_data = { ... }  # existing normal response (lines 200-221)
        return _json_safe_response(resp_data)

    except Exception as primary_error:
        # NEW: attempt to return partial analysis with defaults
        try:
            expr = analyzer._parse(req.expression, req.variable)
            interval = req.interval or analyzer._infer_interval(expr, req.variable)
            a, b = float(interval[0]), float(interval[1])

            resp_data = {
                "original_expr": str(req.expression),
                "recommended_family": "Legendre",       # safe default for finite intervals
                "confidence": "low",
                "recommendation_reason": f"Partial analysis (full probe failed): {str(primary_error)[:200]}",
                "suggested_min_n": 32,
                "suggested_max_n": 128,
                "interval_a": _safe_float(a),
                "interval_b": _safe_float(b),
                "singularities": [],
                "is_periodic": False,
                "derivative_growth": "unknown",
                # C4 fields with safe defaults:
                "interval_type": analyzer._classify_interval(a, b),
                "decay_type": "none",
                "decay_rate": 0.0,
                "hermite_compatible": True,
                "degree_criteria": [f"Analysis degraded: {str(primary_error)[:120]}"],
                "left_singularity_alpha": 0.0,
                "right_singularity_beta": 0.0,
                "has_algebraic_endpoint_singularity": False,
            }
            return _json_safe_response(resp_data)

        except Exception:
            # If even partial analysis fails, return 400
            raise HTTPException(status_code=400, detail=str(primary_error))
```

**Estimated effort:** ~35 lines added to one endpoint

---

### M4: Restore Original Test Expression in Test Suite

**File:** `api_new/comprehensive_grok_suite_Gr1.py`, line 155
**Risk:** None — validation only

After M1–M3 are applied, revert the test suite to use the original expression that was previously avoided:

```python
# BEFORE (current workaround):
{"name": "Analyze_Singularity", "expr": "(1 - x**2)**(-0.3)", "interval": (-0.9, 0.9),
 "expected_has_algebraic_endpoint_singularity": True},

# AFTER:
{"name": "Analyze_Singularity", "expr": "(1 - x)**(-0.3)", "interval": (-1.0, 1.0),
 "expected_has_algebraic_endpoint_singularity": True},
```

---

## Implementation Order & Dependencies

| Phase | Change | Depends On | Effort | Risk |
|-------|--------|------------|--------|------|
| 1 | M1: Guard sub-analyses | None | Small (~12 lines) | Low |
| 2 | M3: Server defensive layer | None (can be done in parallel with M1) | Small (~35 lines) | Medium |
| 3 | M2: mpmath probe evals | None (independent improvement) | Medium (~30 lines) | Low |
| 4 | M4: Restore test expression | M1 and/or M3 completed | Trivial | None |

**Recommended minimum viable fix:** Implement M1 + M3, then M4. This provides both backend-level resilience (M1 prevents cascading failures) and server-level safety net (M3 returns partial results). M2 is an optional accuracy improvement for singularity exponent detection.

---

## Expected Outcome

After implementing M1 + M3:
- `/analyze` with `(1-x)**(-0.3)` on `[-1, 1]` returns HTTP 200 (either full analysis if M1 catches the error, or partial analysis with `"confidence": "low"` if M3 catches it)
- Phase 5 `Analyze_Singularity` test passes without workaround
- No breaking changes to existing behavior for well-behaved expressions

---

## Protected Modules (NOT modified by this plan)

`chebyshev/`, `hermite/`, `laguerre/`, `legendre/`, `api/quadrature_analyzer.py`