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
Done — main_mod4intv.py is ready and compiles clean. Here's what changed:

  Added:

  1. Request import (line 1) — needed for middleware access to raw request body
  2. _sanitize_interval() helper (lines 49–61) — converts any inf/-inf bounds in the interval tuple to None, letting the
     analyzer auto-infer the domain
  3. HTTP middleware (lines 65–82) — intercepts POST requests to /analyze and /integrate, parses the JSON body,
     sanitizes the interval field before Pydantic validation, and rewrites request._body so downstream processing sees
     clean data

  How it works:

  - Client sends (0, float("inf")) → middleware converts to None → analyzer infers semi-infinite domain from expression
    → no JSON serialization errors
  - Client sends (-float("inf"), float("inf")) → same treatment → full infinite interval inferred automatically
  - Finite intervals pass through untouched

  The client can now explicitly state infinity in requests without breaking the server. To test, just swap to the
  modified server:

  py -m main_mod4intv
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


'''
def _sanitize_interval(interval):
    """Convert infinity values in an interval tuple/list to None.
    Accepts numeric inf/-inf or string literals "Infinity" / "-Infinity".
    If any bound is infinite, the whole interval becomes None so the analyzer
    can auto-infer the correct domain from the expression itself."""
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
        if math.isinf(a) or math.isinf(b):
            return None  # let analyzer infer
        return (a, b)
    except (TypeError, ValueError, IndexError):
        return interval
'''


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

# --- Middleware: pre-sanitize incoming JSON bodies ---

@app.middleware("http")
async def sanitize_request_intervals(request: Request, call_next):
    """Intercept POST requests with JSON bodies and convert any inf/-inf values
    in the 'interval' field to None before Pydantic validation.
    This prevents FastAPI from failing on non-JSON-compliant float values."""
    if request.method == "POST" and request.url.path in ("/analyze", "/integrate"):
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


class IntegrationRequest(BaseModel):
    expression: str
    interval: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    variable: str = "x"
    use_mpmath: bool = False

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
        return _json_safe_response(resp_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/integrate")
async def integrate_function(req: IntegrationRequest):
    try:
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
                variable=req.variable
            )
            n = analysis.suggested_max_n
        else:
            n = req.n

        result = analyzer.execute_quadrature(
            req.expression,
            interval=req.interval,
            n=n,
            variable=req.variable,
            use_mpmath=req.use_mpmath
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)