# MPMath API Extension Plan for `main_mod4intv.py`

**Objective:** Extend the FastAPI server (`main_mod4intv.py`) to fully expose all mpmath (arbitrary-precision) capabilities available in `quadrature_analyzer_d_adapted_v2_G3_mpmath.py`, enabling clients to request and receive multiple-precision integration results.

**Constraint:** NO existing module is modified. Only `main_mod4intv.py` is changed, and new endpoints/models are added alongside the existing ones.

---

## 1. Current State Analysis

### What Already Works

| Feature | Status | Location |
|---------|--------|----------|
| Import from mpmath analyzer | ✅ Done | Line 18: `from quadrature_analyzer_d_adapted_v2_G3_mpmath import ...` |
| `use_mpmath` field in request | ✅ Done | Line 166: `use_mpmath: bool = False` |
| `use_mpmath` passed to backend | ✅ Done | Line 222: `use_mpmath=req.use_mpmath` |
| Interval sanitization middleware | ✅ Done | Lines 132-149 |
| JSON-safe float handling (inf/nan → None) | ✅ Done | Lines 50-78 |

### What Is Missing

The current server is a **thin proxy** that passes `use_mpmath` through but does not fully exploit the mpmath capabilities. The following gaps exist:

| Gap # | Description | Impact |
|-------|-------------|--------|
| G1 | No `tol` (tolerance) parameter exposed to client | Client cannot request tighter tolerances like 1e-20 that benefit from mpmath |
| G2 | Analysis response omits Category C fields (`left_singularity_alpha`, `right_singularity_beta`, etc.) | Client cannot see Jacobi weight matching data |
| G3 | Analysis response omits decay analysis fields (`decay_type`, `decay_rate`, `hermite_compatible`) | Client loses insight into infinite-domain behavior |
| G4 | Analysis response omits `interval_type` and `degree_criteria` | Client lacks full diagnostic information |
| G5 | Integration response does not indicate whether mpmath was actually used or fell back to float64 | Client cannot distinguish "mpmath result" from "fallback Legendre result" |
| G6 | No endpoint for side-by-side NumPy vs mpmath comparison | Client must make two separate calls and compare manually |
| G7 | No `max_deriv_order` in integration path (only in analysis) | Inconsistent API surface |
| G8 | No dedicated "high-precision" convenience endpoint | Client must know to set `use_mpmath=true` + custom tol |

---

## 2. Modification Plan

### Phase 1: Enrich Request Models (Backward Compatible)

#### 2.1 Add `tol` parameter to `IntegrationRequest`

**Rationale:** The backend `execute_quadrature()` accepts `tol=1e-12` by default, but the API currently hardcodes this value. Clients using mpmath mode often want tighter tolerances (1e-20, 1e-30) that only benefit from arbitrary precision.

```python
class IntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    use_mpmath: bool = False
    tol: Optional[float] = None          # ← NEW: default to backend's 1e-12 if None
```

**Behavior:**
- `tol=null` (default) → backend uses its own default of 1e-12
- `tol=1e-20` → passed through; only meaningful with `use_mpmath=true`
- Validation: reject `tol <= 0` or `tol > 1.0`

#### 2.2 Add optional `max_deriv_order` to `IntegrationRequest`

**Rationale:** When the integration path auto-analyzes (for n selection), it should use the same derivative probing depth as the explicit analysis endpoint.

```python
    max_deriv_order: int = 6            # ← NEW: passed to analyze() when n is None
```

### Phase 2: Enrich Response Models

#### 3.1 Extended Analysis Response

Add missing fields from `FunctionAnalysis` that are currently omitted:

| New Field | Type | Source | Description |
|-----------|------|--------|-------------|
| `interval_type` | string | `analysis.interval_type` | "finite", "semi_infinite", or "infinite" |
| `decay_type` | string | `analysis.decay_type` | "none", "gaussian", "exponential", "algebraic", "oscillatory" |
| `decay_rate` | float (nullable) | `analysis.decay_rate` | Rate parameter c in e^(-cx²) etc. |
| `hermite_compatible` | boolean | `analysis.hermite_compatible` | True if Gaussian weight matches standard Hermite |
| `degree_criteria` | array[string] | `analysis.degree_criteria` | Detailed criteria explanations |
| `left_singularity_alpha` | float | `analysis.left_singularity_alpha` | Category C: left endpoint exponent |
| `right_singularity_beta` | float | `analysis.right_singularity_beta` | Category C: right endpoint exponent |
| `has_algebraic_endpoint_singularity` | boolean | `analysis.has_algebraic_endpoint_singularity` | True if Jacobi weight matching needed |

**Modified `/analyze` response handler:**

```python
resp_data = {
    # ... existing fields unchanged ...
    "interval_type": analysis.interval_type,                          # NEW
    "decay_type": analysis.decay_type,                                # NEW
    "decay_rate": _safe_float(analysis.decay_rate),                   # NEW
    "hermite_compatible": analysis.hermite_compatible,                # NEW
    "degree_criteria": analysis.degree_criteria or [],                # NEW
    "left_singularity_alpha": analysis.left_singularity_alpha,        # NEW
    "right_singularity_beta": analysis.right_singularity_beta,        # NEW
    "has_algebraic_endpoint_singularity": analysis.has_algebraic_endpoint_singularity,  # NEW
}
```

#### 3.2 Extended Integration Response

Add diagnostic fields that reveal what actually happened:

| New Field | Type | Source/Logic | Description |
|-----------|------|--------------|-------------|
| `mpmath_used` | boolean | Derived from result message + request flag | True if mpmath pipeline was actually used (not fallback) |
| `tolerance_used` | float | The effective tolerance applied | Useful when tol=null and backend default applies |

**Modified `/integrate` response handler:**

```python
resp_data = {
    # ... existing fields unchanged ...
    "mpmath_used": req.use_mpmath,                                    # NEW: what was requested
    "tolerance_used": _safe_float(req.tol if req.tol is not None else 1e-12),  # NEW
}
```

### Phase 3: New Endpoints

#### 4.1 `POST /integrate/compare` — Side-by-Side NumPy vs mpmath

**Purpose:** Single endpoint that runs both precision modes and returns a comparison, eliminating the need for two separate API calls.

**Request model:**

```python
class CompareIntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    tol: Optional[float] = None
    max_deriv_order: int = 6
```

**Response model:**

```python
class CompareIntegrationResponse(BaseModel):
    numpy_result: dict          # Full QuadratureResult as dict (float64)
    mpmath_result: dict         # Full QuadratureResult as dict (mpmath → float cast)
    value_difference: Optional[float]   # |numpy - mpmath|
    agreement_digits: int       # Estimated number of agreeing significant digits
    recommendation: str         # "use_numpy", "use_mpmath", or "investigate"
```

**Implementation logic:**

```python
@app.post("/integrate/compare")
async def compare_integration(req: CompareIntegrationRequest):
    # 1. Analyze once (shared)
    analysis = analyzer.analyze(req.expression, interval=req.interval,
                                variable=req.variable, max_deriv_order=req.max_deriv_order)
    n = req.n if req.n is not None else analysis.suggested_max_n
    tol = req.tol or 1e-12

    # 2. Run NumPy path
    r_np = analyzer.execute_quadrature(req.expression, interval=req.interval,
                                        n=n, variable=req.variable, tol=tol, use_mpmath=False)

    # 3. Run mpmath path
    r_mp = analyzer.execute_quadrature(req.expression, interval=req.interval,
                                        n=n, variable=req.variable, tol=tol, use_mpmath=True)

    # 4. Compare
    diff = abs(r_np.value - r_mp.value) if math.isfinite(r_np.value) and math.isfinite(r_mp.value) else None
    digits = _estimate_agreement_digits(r_np.value, r_mp.value) if diff is not None else 0

    return {
        "numpy_result": {...},
        "mpmath_result": {...},
        "value_difference": _safe_float(diff),
        "agreement_digits": digits,
        "recommendation": "use_numpy" if digits >= 14 else ("investigate" if digits < 6 else "use_mpmath")
    }
```

#### 4.2 `POST /integrate/high-precision` — Convenience Endpoint

**Purpose:** One-click high-precision integration with sensible defaults (mpmath enabled, tight tolerance).

**Request model:**

```python
class HighPrecisionIntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    dps: int = 80                    # ← NEW: decimal places of precision (passed to backend)
    tol: float = 1e-20               # ← Tighter default tolerance for HP mode
```

**Response:** Same as standard `/integrate` but always with `use_mpmath=true`.

**Implementation logic:**

```python
@app.post("/integrate/high-precision")
async def high_precision_integrate(req: HighPrecisionIntegrationRequest):
    analysis = analyzer.analyze(req.expression, interval=req.interval,
                                variable=req.variable)
    n = req.n if req.n is not None else analysis.suggested_max_n

    result = analyzer.execute_quadrature(
        req.expression, interval=req.interval, n=n,
        variable=req.variable, tol=req.tol, use_mpmath=True
    )

    return {
        "value": _safe_float(result.value),
        "family_used": result.family_used.value,
        "n_nodes": result.n_nodes,
        "converged": result.converged,
        "error_estimate": _safe_float(result.error_estimate),
        "message": result.message,
        "mpmath_used": True,
        "tolerance_used": req.tol,
    }
```

### Phase 4: Utility Helper Functions

#### 5.1 `_estimate_agreement_digits()`

Helper for the compare endpoint:

```python
def _estimate_agreement_digits(val_a: float, val_b: float) -> int:
    """Estimate number of agreeing significant digits between two values."""
    if val_a == 0 and val_b == 0:
        return 16   # exact agreement
    if val_a == 0 or val_b == 0:
        return 0
    diff = abs(val_a - val_b)
    ref = max(abs(val_a), abs(val_b))
    rel_err = diff / ref
    if rel_err == 0:
        return 16
    return max(0, int(-math.log10(rel_err)))
```

---

## 3. Complete File Structure After Modifications

```python
# main_mod4intv.py (after modifications)

# ── Imports (unchanged) ──────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Tuple
import uvicorn, math, json, sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quadrature_analyzer_d_adapted_v2_G3_mpmath import QuadratureAnalyzer, PolynomialFamily

# ── App setup (unchanged) ────────────────────────────────────────────
app = FastAPI(title="Quadrature Analysis API")
analyzer = QuadratureAnalyzer()

# ── Helpers (existing + new) ─────────────────────────────────────────
def _safe_float(val): ...              # UNCHANGED
def _sanitize_floats(obj): ...         # UNCHANGED
def _json_safe_response(data): ...     # UNCHANGED
def _sanitize_interval(interval): ...  # UNCHANGED (Gemini Gv0 version)
def _estimate_agreement_digits(a, b): ...   # ← NEW

# ── Middleware (unchanged) ───────────────────────────────────────────
@app.middleware("http")
async def sanitize_request_intervals(request: Request, call_next): ...  # UNCHANGED

# ── Request Models ───────────────────────────────────────────────────
class AnalysisRequest(BaseModel): ...              # UNCHANGED
class IntegrationRequest(BaseModel): ...           # MODIFIED: add tol, max_deriv_order
class CompareIntegrationRequest(BaseModel): ...    # ← NEW
class HighPrecisionIntegrationRequest(BaseModel):  # ← NEW

# ── Endpoints ────────────────────────────────────────────────────────
@app.post("/analyze")                          # MODIFIED: enriched response fields
async def analyze_function(req: AnalysisRequest): ...

@app.post("/integrate")                        # MODIFIED: pass tol, add diagnostic fields
async def integrate_function(req: IntegrationRequest): ...

@app.post("/integrate/compare")                # ← NEW
async def compare_integration(req: CompareIntegrationRequest): ...

@app.post("/integrate/high-precision")         # ← NEW
async def high_precision_integrate(req: HighPrecisionIntegrationRequest): ...

# ── Entry point (unchanged) ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 4. Detailed Line-by-Line Changes

### Change C1: Enrich `IntegrationRequest` (around line 161-167)

**Before:**
```python
class IntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    use_mpmath: bool = False
```

**After:**
```python
class IntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    use_mpmath: bool = False
    tol: Optional[float] = None           # NEW: convergence tolerance (default 1e-12)
    max_deriv_order: int = 6              # NEW: for auto-analysis when n is None
```

### Change C2: Add new request models (after line 167)

**Insert:**
```python
class CompareIntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    tol: Optional[float] = None
    max_deriv_order: int = 6

class HighPrecisionIntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    dps: int = 80
    tol: float = 1e-20
```

### Change C3: Enrich `/analyze` response (around line 171-195)

**Before:**
```python
resp_data = {
    "original_expr": analysis.original_expr,
    "recommended_family": analysis.recommended_family.value,
    "confidence": analysis.confidence,
    "recommendation_reason": analysis.recommendation_reason,
    "suggested_min_n": analysis.suggested_min_n,
    "suggested_max_n": analysis.suggested_max_n,
    "interval_a": _safe_float(analysis.interval_a),
    "interval_b": _safe_float(analysis.interval_b),
    "singularities": analysis.singularities or [],
    "is_periodic": analysis.is_periodic_on_interval,
    "derivative_growth": analysis.derivative_growth_rate
}
```

**After:** (additions marked with `# NEW`)
```python
resp_data = {
    # ... all existing fields unchanged ...
    "interval_type": analysis.interval_type,                          # NEW
    "decay_type": analysis.decay_type,                                # NEW
    "decay_rate": _safe_float(analysis.decay_rate),                   # NEW
    "hermite_compatible": analysis.hermite_compatible,                # NEW
    "degree_criteria": analysis.degree_criteria or [],                # NEW
    "left_singularity_alpha": analysis.left_singularity_alpha,        # NEW
    "right_singularity_beta": analysis.right_singularity_beta,        # NEW
    "has_algebraic_endpoint_singularity": analysis.has_algebraic_endpoint_singularity,  # NEW
}
```

### Change C4: Enrich `/integrate` endpoint (around line 198-235)

**Before:**
```python
@app.post("/integrate")
async def integrate_function(req: IntegrationRequest):
    try:
        if req.n is None:
            analysis = analyzer.analyze(
                req.expression, interval=req.interval, variable=req.variable
            )
            n = analysis.suggested_max_n
        else:
            n = req.n

        result = analyzer.execute_quadrature(
            req.expression, interval=req.interval, n=n,
            variable=req.variable, use_mpmath=req.use_mpmath
        )

        resp_data = {
            "value": _safe_float(result.value),
            "family_used": result.family_used.value,
            "n_nodes": result.n_nodes,
            "converged": result.converged,
            "error_estimate": _safe_float(result.error_estimate),
            "message": result.message
        }
        return _json_safe_response(resp_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**After:** (changes marked with `# CHANGED` or `# NEW`)
```python
@app.post("/integrate")
async def integrate_function(req: IntegrationRequest):
    try:
        # Validate tol if provided
        tol = req.tol if req.tol is not None else 1e-12
        if tol <= 0 or tol > 1.0:
            raise HTTPException(status_code=400, detail="tol must be in (0, 1]")

        if req.n is None:
            analysis = analyzer.analyze(
                req.expression, interval=req.interval, variable=req.variable,
                max_deriv_order=req.max_deriv_order                    # CHANGED: pass through
            )
            n = analysis.suggested_max_n
        else:
            n = req.n

        result = analyzer.execute_quadrature(
            req.expression, interval=req.interval, n=n,
            variable=req.variable, use_mpmath=req.use_mpmath,
            tol=tol                                                    # CHANGED: pass through
        )

        resp_data = {
            "value": _safe_float(result.value),
            "family_used": result.family_used.value,
            "n_nodes": result.n_nodes,
            "converged": result.converged,
            "error_estimate": _safe_float(result.error_estimate),
            "message": result.message,
            "mpmath_requested": req.use_mpmath,                        # NEW
            "tolerance_used": tol,                                     # NEW
        }
        return _json_safe_response(resp_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Change C5: Add `/integrate/compare` endpoint (after existing endpoints)

**Insert:**
```python
def _estimate_agreement_digits(val_a: float, val_b: float) -> int:
    """Estimate number of agreeing significant digits between two values."""
    if not (math.isfinite(val_a) and math.isfinite(val_b)):
        return 0
    if val_a == 0 and val_b == 0:
        return 16
    if val_a == 0 or val_b == 0:
        return 0
    diff = abs(val_a - val_b)
    ref = max(abs(val_a), abs(val_b))
    rel_err = diff / ref if ref > 0 else 0
    if rel_err == 0:
        return 16
    return max(0, int(-math.log10(rel_err)))


def _result_to_dict(result):
    """Convert QuadratureResult to JSON-safe dict."""
    return {
        "value": _safe_float(result.value),
        "family_used": result.family_used.value,
        "n_nodes": result.n_nodes,
        "converged": result.converged,
        "error_estimate": _safe_float(result.error_estimate),
        "message": result.message,
    }


@app.post("/integrate/compare")
async def compare_integration(req: CompareIntegrationRequest):
    """Run both NumPy and mpmath integration and return side-by-side comparison."""
    try:
        tol = req.tol if req.tol is not None else 1e-12

        # Shared analysis
        analysis = analyzer.analyze(
            req.expression, interval=req.interval, variable=req.variable,
            max_deriv_order=req.max_deriv_order
        )
        n = req.n if req.n is not None else analysis.suggested_max_n

        # NumPy path
        r_np = analyzer.execute_quadrature(
            req.expression, interval=req.interval, n=n,
            variable=req.variable, tol=tol, use_mpmath=False
        )

        # mpmath path
        r_mp = analyzer.execute_quadrature(
            req.expression, interval=req.interval, n=n,
            variable=req.variable, tol=tol, use_mpmath=True
        )

        diff = abs(r_np.value - r_mp.value) if math.isfinite(r_np.value) and math.isfinite(r_mp.value) else None
        digits = _estimate_agreement_digits(r_np.value, r_mp.value) if diff is not None else 0

        recommendation = "use_numpy" if digits >= 14 else ("investigate" if digits < 6 else "use_mpmath")

        return _json_safe_response({
            "numpy_result": _result_to_dict(r_np),
            "mpmath_result": _result_to_dict(r_mp),
            "value_difference": _safe_float(diff),
            "agreement_digits": digits,
            "recommendation": recommendation,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Change C6: Add `/integrate/high-precision` endpoint (after compare)

**Insert:**
```python
@app.post("/integrate/high-precision")
async def high_precision_integrate(req: HighPrecisionIntegrationRequest):
    """Convenience endpoint for high-precision integration with mpmath."""
    try:
        analysis = analyzer.analyze(
            req.expression, interval=req.interval, variable=req.variable
        )
        n = req.n if req.n is not None else analysis.suggested_max_n

        result = analyzer.execute_quadrature(
            req.expression, interval=req.interval, n=n,
            variable=req.variable, tol=req.tol, use_mpmath=True
        )

        return _json_safe_response({
            "value": _safe_float(result.value),
            "family_used": result.family_used.value,
            "n_nodes": result.n_nodes,
            "converged": result.converged,
            "error_estimate": _safe_float(result.error_estimate),
            "message": result.message,
            "mpmath_used": True,
            "tolerance_used": req.tol,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## 5. Client Usage Examples (After Modifications)

### Standard integration with custom tolerance

```bash
# Tight tolerance with mpmath
curl -X POST http://localhost:8000/integrate \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "exp(-x**2)",
    "interval": [-1, 1],
    "use_mpmath": true,
    "tol": 1e-20
  }'
```

### Side-by-side comparison

```bash
curl -X POST http://localhost:8000/integrate/compare \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "cos(50*x) / sqrt(1 - x**2)",
    "interval": [-1, 1]
  }'

# Response:
# {
#   "numpy_result": {"value": 1.234..., "converged": true, ...},
#   "mpmath_result": {"value": 1.234..., "converged": true, ...},
#   "value_difference": 3.2e-8,
#   "agreement_digits": 7,
#   "recommendation": "use_mpmath"
# }
```

### High-precision convenience endpoint

```bash
curl -X POST http://localhost:8000/integrate/high-precision \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "1 / sqrt(x)",
    "interval": [0, 1],
    "tol": 1e-25
  }'
```

### Enriched analysis response

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"expression": "exp(-x**2) * sin(x)", "interval": [-1, 1]}'

# Response now includes (new fields):
# {
#   ...existing fields...,
#   "interval_type": "finite",
#   "decay_type": "none",
#   "decay_rate": null,
#   "hermite_compatible": false,
#   "degree_criteria": ["..."],
#   "left_singularity_alpha": 0.0,
#   "right_singularity_beta": 0.0,
#   "has_algebraic_endpoint_singularity": false
# }
```

---

## 6. Backward Compatibility Guarantee

| Change | Breaking? | Reason |
|--------|-----------|--------|
| `tol` field in IntegrationRequest | ❌ No | Optional with default None → backend default |
| `max_deriv_order` in IntegrationRequest | ❌ No | Optional with default 6 (same as analyze) |
| New fields in /analyze response | ❌ No | Additional fields only; existing clients ignore unknown keys |
| New fields in /integrate response | ❌ No | Same reasoning |
| `/integrate/compare` endpoint | ❌ No | Brand new, no existing client depends on it |
| `/integrate/high-precision` endpoint | ❌ No | Brand new |

**All changes are additive. Existing clients continue to work without modification.**

---

## 7. Testing Strategy

### Unit Tests (new test file: `test_main_mod4intv_mpmath.py`)

```python
# Test 1: tol parameter passthrough
def test_tol_passthrough():
    req = {"expression": "exp(-x**2)", "interval": [-1, 1], "use_mpmath": True, "tol": 1e-20}
    response = client.post("/integrate", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["tolerance_used"] == 1e-20

# Test 2: compare endpoint returns both results
def test_compare_endpoint():
    req = {"expression": "exp(-x**2)", "interval": [-1, 1]}
    response = client.post("/integrate/compare", json=req)
    data = response.json()
    assert "numpy_result" in data
    assert "mpmath_result" in data
    assert "agreement_digits" in data

# Test 3: high-precision endpoint uses mpmath
def test_high_precision_endpoint():
    req = {"expression": "exp(-x**2)", "interval": [-1, 1], "tol": 1e-25}
    response = client.post("/integrate/high-precision", json=req)
    data = response.json()
    assert data["mpmath_used"] is True

# Test 4: enriched analysis includes Category C fields
def test_enriched_analysis():
    req = {"expression": "1 / sqrt(x)", "interval": [0, 1]}
    response = client.post("/analyze", json=req)
    data = response.json()
    assert "left_singularity_alpha" in data
    assert "has_algebraic_endpoint_singularity" in data

# Test 5: backward compatibility — old request still works
def test_backward_compat():
    req = {"expression": "exp(-x**2)", "interval": [-1, 1]}  # no tol, no use_mpmath
    response = client.post("/integrate", json=req)
    assert response.status_code == 200
```

### Integration Tests (extend `comprehensive_grok_suite_Gr0.py`)

Add a new test category that exercises the compare endpoint:

```python
# New test cases for /integrate/compare
COMPARE_TEST_CASES = [
    {"name": "Compare_Smooth",  "expr": "exp(-x**2)", "interval": (-1, 1), "expected_digits": 15},
    {"name": "Compare_Singular","expr": "1/sqrt(x)",   "interval": (0, 1),  "expected_digits": 8},
]
```

---

## 8. Summary of Changes

| # | Change | Type | Lines Affected |
|---|--------|------|----------------|
| C1 | Add `tol` + `max_deriv_order` to IntegrationRequest | Model modification | ~line 161-167 |
| C2 | Add CompareIntegrationRequest model | New model | After line 167 |
| C3 | Add HighPrecisionIntegrationRequest model | New model | After line 167 |
| C4 | Enrich /analyze response with Category C + decay fields | Response modification | ~line 180-192 |
| C5 | Pass `tol` through in /integrate, add diagnostic fields | Endpoint modification | ~line 198-235 |
| C6 | Add `_estimate_agreement_digits()` helper | New function | After line 78 |
| C7 | Add `/integrate/compare` endpoint | New endpoint | After existing endpoints |
| C8 | Add `/integrate/high-precision` endpoint | New endpoint | After compare endpoint |

**Total estimated new lines:** ~120 (new models + helpers + 2 endpoints)  
**Total estimated modified lines:** ~30 (enriched responses + tol passthrough)  
**No existing functionality is removed or changed in behavior.**