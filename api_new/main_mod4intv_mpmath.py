from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Tuple
import uvicorn
import math
import json

# Import the analyzer from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from quadrature_analyzer import QuadratureAnalyzer, PolynomialFamily
# from quadrature_analyzer_hermite_improved import QuadratureAnalyzer, PolynomialFamily
# from quadrature_analyzer_laguerre_fixed import QuadratureAnalyzer, PolynomialFamily
# from quadrature_analyzer_d_adapted import QuadratureAnalyzer, PolynomialFamily
# from quadrature_analyzer_d_adapted_v2 import QuadratureAnalyzer, PolynomialFamily
# from quadrature_analyzer_d_adapted_v2_G33 import QuadratureAnalyzer, PolynomialFamily
from quadrature_analyzer_d_adapted_v2_G3_mpmath import QuadratureAnalyzer, PolynomialFamily

"""
main_mod4intv_mpmath.py — MPMath API Extension (based on main_mod4intv.py)

Changes applied from MPMATH_API_EXTENSION_PLAN.md:
  C1: Added tol + max_deriv_order to IntegrationRequest
  C2: Added CompareIntegrationRequest model
  C3: Added HighPrecisionIntegrationRequest model
  C4: Enriched /analyze response with Category C + decay fields
  C5: Pass tol through in /integrate, add diagnostic fields
  C6: Added _estimate_agreement_digits() helper
  C7: Added /integrate/compare endpoint
  C8: Added /integrate/high-precision endpoint

No existing module (chebyshev, hermite, laguerre, legendre) was modified.
"""

app = FastAPI(title="Quadrature Analysis API")
analyzer = QuadratureAnalyzer()

# --- Helpers (defined FIRST so they are available everywhere) ---

def _safe_float(val):
    """Convert a float to JSON-safe value. inf/nan become None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None

def _sanitize_floats(obj):
    """Recursively replace all inf/nan floats with None in any nested structure."""
    if isinstance(obj, float):
        return _safe_float(obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_floats(item) for item in obj]
    return obj

def _json_safe_response(data):
    """Sanitize all floats and return a dict ready for FastAPI JSONResponse."""
    sanitized = _sanitize_floats(data)
    # Double-check: if anything is still inf/nan after sanitization, force to None
    raw_json = json.dumps(sanitized)
    return json.loads(raw_json)  # This will raise if anything bad remains


# BEGIN Gemini Gv0
def _sanitize_interval(interval):
    """Convert infinity values in an interval tuple/list to standard python floats.
    Accepts numeric inf/-inf or string literals "Infinity" / "-Infinity".
    (No longer converts them to None, preserving explicit user intent)."""
    if interval is None:
        return None
    try:
        def _parse_bound(v):
            if isinstance(v, str):
                v = v.strip()
                if v in ("Infinity", "inf"):
                    return float("inf")
                if v in ("-Infinity", "-inf"):
                    return float("-inf")
            return float(v)

        a, b = _parse_bound(interval[0]), _parse_bound(interval[1])
        # REMOVED: the aggressive conversion to None
        return (a, b)
    except (TypeError, ValueError, IndexError):
        return interval
# END Gemini Gv0


# C6: NEW helper — estimate agreeing significant digits between two values
def _estimate_agreement_digits(val_a: float, val_b: float) -> int:
    """Estimate number of agreeing significant digits between two values."""
    if not (math.isfinite(val_a) and math.isfinite(val_b)):
        return 0
    if val_a == 0 and val_b == 0:
        return 16   # exact agreement at float64 limit
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


# --- Middleware: pre-sanitize incoming JSON bodies ---

@app.middleware("http")
async def sanitize_request_intervals(request: Request, call_next):
    """Intercept POST requests with JSON bodies and convert any inf/-inf values
    in the 'interval' field to None before Pydantic validation.
    This prevents FastAPI from failing on non-JSON-compliant float values."""
    # Extended path list to cover new endpoints
    if request.method == "POST" and request.url.path.startswith(("/analyze", "/integrate")):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.json()
                if "interval" in body and body["interval"] is not None:
                    body["interval"] = _sanitize_interval(body["interval"])
                raw = json.dumps(body).encode("utf-8")
                request._body = raw
            except Exception:
                pass  # if parsing fails, let normal validation handle it
    response = await call_next(request)
    return response

# --- Request/Response Models ---


class AnalysisRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    variable: str = "x"
    max_deriv_order: int = 6


# C1: Enriched IntegrationRequest with tol and max_deriv_order
class IntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    use_mpmath: bool = False
    tol: Optional[float] = None              # C1 NEW: convergence tolerance (default 1e-12)
    max_deriv_order: int = 6                 # C1 NEW: for auto-analysis when n is None


# C2: CompareIntegrationRequest — side-by-side NumPy vs mpmath
class CompareIntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    tol: Optional[float] = None
    max_deriv_order: int = 6


# C3: HighPrecisionIntegrationRequest — convenience endpoint for mpmath
class HighPrecisionIntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    dps: int = 80                            # decimal places of precision (informational)
    tol: float = 1e-20                       # tighter default tolerance for HP mode

# --- Endpoints ---


@app.post("/analyze")
async def analyze_function(req: AnalysisRequest):
    try:
        analysis = analyzer.analyze(
            req.expression,
            interval=req.interval,
            variable=req.variable,
            max_deriv_order=req.max_deriv_order
        )
        # C4: Enriched response with Category C + decay fields
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
            "derivative_growth": analysis.derivative_growth_rate,
            # C4 NEW fields below:
            "interval_type": analysis.interval_type,
            "decay_type": analysis.decay_type,
            "decay_rate": _safe_float(analysis.decay_rate),
            "hermite_compatible": analysis.hermite_compatible,
            "degree_criteria": analysis.degree_criteria or [],
            "left_singularity_alpha": analysis.left_singularity_alpha,
            "right_singularity_beta": analysis.right_singularity_beta,
            "has_algebraic_endpoint_singularity": analysis.has_algebraic_endpoint_singularity,
        }
        return _json_safe_response(resp_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/integrate")
async def integrate_function(req: IntegrationRequest):
    try:
        # C5: Validate and resolve tolerance
        tol = req.tol if req.tol is not None else 1e-12
        if tol <= 0 or tol > 1.0:
            raise HTTPException(status_code=400, detail="tol must be in (0, 1]")

        # If n is not provided, analyze first to get suggested max_n.
        # IMPORTANT: must use suggested_max_n (not min_n) to match the direct
        # backend path in execute_quadrature() which defaults to suggested_max_n.
        # Using min_n here causes Category A convergence boundary failures where
        # Direct and API compute with different node counts, producing mismatched
        # values or divergent convergence flags at tol=1e-12 thresholds.
        if req.n is None:
            analysis = analyzer.analyze(
                req.expression,
                interval=req.interval,
                variable=req.variable,
                max_deriv_order=req.max_deriv_order    # C5 CHANGED: pass through
            )
            n = analysis.suggested_max_n
        else:
            n = req.n

        result = analyzer.execute_quadrature(
            req.expression,
            interval=req.interval,
            n=n,
            variable=req.variable,
            use_mpmath=req.use_mpmath,
            tol=tol                                     # C5 CHANGED: pass through
        )

        resp_data = {
            "value": _safe_float(result.value),
            "family_used": result.family_used.value,
            "n_nodes": result.n_nodes,
            "converged": result.converged,
            "error_estimate": _safe_float(result.error_estimate),
            "message": result.message,
            # C5 NEW diagnostic fields:
            "mpmath_requested": req.use_mpmath,
            "tolerance_used": tol,
        }
        return _json_safe_response(resp_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# C7: NEW endpoint — side-by-side NumPy vs mpmath comparison
@app.post("/integrate/compare")
async def compare_integration(req: CompareIntegrationRequest):
    """Run both NumPy and mpmath integration and return a side-by-side comparison.

    This eliminates the need for two separate API calls when you want to verify
    whether float64 precision is sufficient or if arbitrary-precision mpmath is needed.
    """
    try:
        tol = req.tol if req.tol is not None else 1e-12
        if tol <= 0 or tol > 1.0:
            raise HTTPException(status_code=400, detail="tol must be in (0, 1]")

        # Shared analysis (done once for both paths)
        analysis = analyzer.analyze(
            req.expression, interval=req.interval, variable=req.variable,
            max_deriv_order=req.max_deriv_order
        )
        n = req.n if req.n is not None else analysis.suggested_max_n

        # NumPy path (float64)
        r_np = analyzer.execute_quadrature(
            req.expression, interval=req.interval, n=n,
            variable=req.variable, tol=tol, use_mpmath=False
        )

        # mpmath path (arbitrary precision)
        r_mp = analyzer.execute_quadrature(
            req.expression, interval=req.interval, n=n,
            variable=req.variable, tol=tol, use_mpmath=True
        )

        # Compute agreement metrics
        diff = abs(r_np.value - r_mp.value) if math.isfinite(r_np.value) and math.isfinite(r_mp.value) else None
        digits = _estimate_agreement_digits(r_np.value, r_mp.value) if diff is not None else 0

        # Recommendation logic:
        #   >= 14 agreeing digits → float64 is sufficient (use_numpy)
        #    <  6 agreeing digits → results differ significantly (investigate)
        #   6-13 agreeing digits  → mpmath provides better accuracy (use_mpmath)
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


# C8: NEW endpoint — convenience high-precision integration with mpmath
@app.post("/integrate/high-precision")
async def high_precision_integrate(req: HighPrecisionIntegrationRequest):
    """Convenience endpoint for high-precision integration using mpmath.

    Always uses use_mpmath=True and a tight default tolerance (1e-20).
    Ideal for cases where maximum accuracy is desired without configuring
    individual parameters.
    """
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)