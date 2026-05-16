from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple
import uvicorn
import math
import json

# Import the analyzer from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quadrature_analyzer import QuadratureAnalyzer, PolynomialFamily

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
        # If n is not provided, analyze first to get suggested min_n
        if req.n is None:
            analysis = analyzer.analyze(
                req.expression,
                interval=req.interval,
                variable=req.variable
            )
            n = analysis.suggested_min_n
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
