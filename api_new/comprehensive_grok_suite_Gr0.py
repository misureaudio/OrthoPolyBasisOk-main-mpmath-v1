"""Comprehensive Test Suite: Backend vs Middleware Fidelity Check.

Validates that the FastAPI middleware (main_mod4intv.py) faithfully proxies
requests to the quadrature_analyzer backend by comparing direct backend calls
(Path A — Ground Truth) against HTTP API calls (Path B — Proxy under test).

Requirements:
    - Server must be `main_mod4intv.py` (has interval-sanitization middleware).
      Start with:  python -m api.main_mod4intv
    - Alternatively set QUADRATURE_API_URL env var to point to any compatible server.

Tolerance note:
    - This suite uses tol=1e-9 for value comparisons, intentionally more lenient
      than the engine's internal convergence tolerance of tol=1e-12.

Protected modules (NEVER modified by this plan):
    legendre/, chebyshev/, hermite/, laguerre/, api/quadrature_analyzer.py
"""

import math
import os
import sys
from prettytable import PrettyTable
import requests

# ---------------------------------------------------------------------------
# Import the backend directly for Ground Truth generation
# FIX (Phase 1): Use project root on sys.path to match main.py / main_mod4intv.py
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from quadrature_analyzer import QuadratureAnalyzer
# from quadrature_analyzer_hermite_improved import QuadratureAnalyzer
# from quadrature_analyzer_laguerre_fixed import QuadratureAnalyzer
# from quadrature_analyzer_d_adapted import QuadratureAnalyzer
# from quadrature_analyzer_d_adapted_v2 import QuadratureAnalyzer
# from quadrature_analyzer_d_adapted_v2_G3 import QuadratureAnalyzer
from quadrature_analyzer_d_adapted_v2_G3_mpmath import QuadratureAnalyzer

# Configurable API URL (env var overrides default)
API_URL = os.environ.get("QUADRATURE_API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# 1. CENTRAL TEST REGISTRY
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ====================== LEGENDRE ======================
    # Polynomial exactness (should be exact up to degree 2n-1)
    {"name": "Leg_Poly_Deg5",      "expr": "x**5 - 3*x**2 + 1",          "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Poly_Deg10",     "expr": "x**10 + 2*x**7 - x**3 + 5",  "interval": (-1.0, 1.0), "target_family": "Legendre"},

    # Smooth entire functions (excellent for high n / high dps)
    {"name": "Leg_Exp_Mild",       "expr": "exp(0.5*x)",                 "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Exp_Strong",     "expr": "exp(1.8*x)",                 "interval": (-1.0, 1.0), "target_family": "Legendre"},

    # Oscillatory (tests cancellation at large n)
    {"name": "Leg_Osc_Moderate",   "expr": "cos(8*x)",                   "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Osc_High",       "expr": "sin(40*x) + cos(25*x)",      "interval": (-1.0, 1.0), "target_family": "Legendre"},

    # Mild endpoint singularities (tests clustering)
    {"name": "Leg_Sing_Alg",       "expr": "(1 - x**2)**(-0.3) * exp(0.3*x)", "interval": (-1.0, 1.0), "target_family": "Legendre"},

    # ====================== CHEBYSHEV ======================
    # Natural for Chebyshev weight 1/sqrt(1-x²)
    {"name": "Cheb_Tk_Exact",      "expr": "cos(12 * acos(x))",          "interval": (-1.0, 1.0), "target_family": "Chebyshev"},
    {"name": "Cheb_Exp",           "expr": "exp(0.7*x)",                 "interval": (-1.0, 1.0), "target_family": "Chebyshev"},
    {"name": "Cheb_Osc_Weighted",  "expr": "cos(20*x) / sqrt(1 - x**2)", "interval": (-1.0, 1.0), "target_family": "Chebyshev"},
    {"name": "Cheb_Sing_Beta",     "expr": "(1 - x**2)**(-0.4) * cos(5*x)", "interval": (-1.0, 1.0), "target_family": "Chebyshev"},

    # ====================== HERMITE ======================
    # Weight e^{-x²}
    {"name": "Herm_Gauss_Exact",   "expr": "exp(-x**2)",                 "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Herm_Perturbed",     "expr": "exp(-0.05*x**2 + 0.8*x)",    "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Herm_Osc_Gauss",     "expr": "cos(15*x) * exp(-0.08*x**2)", "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Herm_Poly_Gauss",    "expr": "(x**6 + 2*x**4 - 3*x**2) * exp(-x**2)", "interval": (-math.inf, math.inf), "target_family": "Hermite"},

    # ====================== LAGUERRE ======================
    # Weight x^α e^{-x}, default α=0
    {"name": "Lag_Exp_Exact",      "expr": "exp(-0.6*x)",                "interval": (0.0, math.inf), "target_family": "Laguerre"},
    {"name": "Lag_Poly_Exp",       "expr": "x**5 * exp(-1.2*x)",         "interval": (0.0, math.inf), "target_family": "Laguerre"},
    {"name": "Lag_Osc_Decay",      "expr": "cos(12*x) * exp(-0.7*x)",    "interval": (0.0, math.inf), "target_family": "Laguerre"},
    {"name": "Lag_Log_Sing",       "expr": "log(1 + x) * exp(-0.5*x)",   "interval": (0.0, math.inf), "target_family": "Laguerre"},

    # ====================== CROSS-FAMILY & STRESS TESTS ======================
    {"name": "Cross_1over1pX4",    "expr": "1/(1 + x**4)",               "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Stress_RapidOsc",    "expr": "sin(120*x)",                 "interval": (0.0, 1.0), "target_family": "Legendre"},
    {"name": "Stress_NarrowPeak",  "expr": "exp(-1/(1 - x**2 + 1e-8))",  "interval": (-0.999, 0.999), "target_family": "Legendre"},
    {"name": "Stress_HighFreqLag", "expr": "sin(50*x) * exp(-0.9*x)",    "interval": (0.0, math.inf), "target_family": "Laguerre"},

    # Additional polynomial + weight interaction
    {"name": "Leg_HighPoly",       "expr": "x**15 - 4*x**9 + 2*x**4",    "interval": (-1.0, 1.0), "target_family": "Legendre"},
]


# ---------------------------------------------------------------------------
# Parameterized generator (adds more cases dynamically for sweeps)
# ---------------------------------------------------------------------------
def generate_parametric_cases():
    """Append more cases programmatically — useful for sweeps."""
    extra = []
    for omega in [5, 25, 80]:
        extra.append({
            "name": f"Leg_Osc_ω{omega}",
            "expr": f"cos({omega}*x)",
            "interval": (-1.0, 1.0),
            "target_family": "Legendre"
        })
    for c in [0.3, 1.1, 2.5]:
        extra.append({
            "name": f"Lag_Exp_c{c}",
            "expr": f"exp(-{c}*x)",
            "interval": (0.0, math.inf),
            "target_family": "Laguerre"
        })
    return extra


TEST_CASES.extend(generate_parametric_cases())


# ---------------------------------------------------------------------------
# Phase 3: Extended test cases exercising non-default API parameters
# ---------------------------------------------------------------------------

EXTENDED_TEST_CASES = [
    # Explicit n parameter in /integrate
    {"name": "Leg_ExplicitN16",    "expr": "x**2",                       "interval": (-1.0, 1.0), "target_family": "Legendre", "integrate_n": 16},
    {"name": "Lag_ExplicitN32",    "expr": "exp(-x)",                    "interval": (0.0, math.inf), "target_family": "Laguerre", "integrate_n": 32},

    # use_mpmath=True high-precision mode
    {"name": "Herm_HighPrec",      "expr": "exp(-x**2)",                 "interval": (-math.inf, math.inf), "target_family": "Hermite", "use_mpmath": True},
]


# ---------------------------------------------------------------------------
# Phase 3: Error-handling test cases
# ---------------------------------------------------------------------------

ERROR_TEST_CASES = [
    {"name": "Err_InvalidExpr",    "expr": "xyz_invalid()",              "interval": (-1.0, 1.0), "expect_error": True},
]


# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------

def _format_api_interval(interval):
    """Formats float(inf) to strings for JSON payload testing."""
    if interval is None:
        return None
    a, b = interval

    def map_val(v):
        if isinstance(v, str):
            return v
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    return (map_val(a), map_val(b))


def float_eq(a, b, tol=1e-9):
    """Compare two numeric values with NaN/None awareness.

    Handles Backend NaN vs API None (due to strict JSON sanitization).
    Uses tol=1e-9 which is intentionally more lenient than the engine's
    internal convergence tolerance of tol=1e-12.
    """
    a_is_nan = isinstance(a, float) and math.isnan(a)
    b_is_nan = isinstance(b, float) and math.isnan(b)

    if a_is_nan and b is None:
        return True
    if b_is_nan and a is None:
        return True
    if a_is_nan and b_is_nan:
        return True

    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    return abs(a - b) < tol


def values_match(a, b, tol=1e-9):
    """Compares two values, properly handling strings, floats, NaN, and None."""
    # If either is an error string, they must match exactly
    if isinstance(a, str) or isinstance(b, str):
        return a == b

    # Handle Backend NaN vs API None (due to strict JSON sanitization)
    a_is_nan = isinstance(a, float) and math.isnan(a)
    b_is_nan = isinstance(b, float) and math.isnan(b)

    if a_is_nan and b is None:
        return True
    if b_is_nan and a is None:
        return True
    if a_is_nan and b_is_nan:
        return True

    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # Both are numeric floats; compare with tolerance
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return a == b


# ---------------------------------------------------------------------------
# Phase 1: Server health check
# ---------------------------------------------------------------------------

def check_server():
    """Verify API is reachable and handles infinity intervals correctly.

    Returns:
        (ok: bool, message: str) — ok=True means tests can proceed normally.
                                   ok=False with a warning means some tests may be skipped.
    """
    try:
        res = requests.post(
            f"{API_URL}/integrate",
            json={
                "expression": "exp(-x**2)",
                "interval": ["-Infinity", "Infinity"]  # test string infinity round-trip
            },
            timeout=5
        )
        if res.status_code == 200:
            data = res.json()
            val = data.get("value")
            if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                return True, "OK — server handles infinity intervals"
            else:
                return True, "WARNING — server returned value=None for infinite interval (may lack middleware)"
        else:
            detail = ""
            try:
                detail = res.json().get("detail", "")[:60]
            except Exception:
                pass
            return False, f"HTTP {res.status_code}{': ' + detail if detail else ''}"

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API server — start main_mod4intv.py first"
    except requests.exceptions.Timeout:
        return False, "Server request timed out (>5s)"


# ---------------------------------------------------------------------------
# 3. TEST RUNNER
# ---------------------------------------------------------------------------

def run_comparison():
    analyzer = QuadratureAnalyzer()

    # --- Pre-flight health check (Phase 1) ---
    server_ok, server_msg = check_server()
    print(f"API Server [{API_URL}]: {server_msg}")
    if not server_ok:
        print("⚠️  Server unreachable. All API comparisons will fail.\n")

    # --- Main comparison table (Phase 2: expanded columns) ---
    table = PrettyTable()
    table.field_names = [
        "Test", "Family(D/A)", "Conv(D/A)",
        "Value (Direct)", "Value (API)", "ErrEst (API)", "Match?"
    ]
    table.align = "l"

    passed_fidelity = 0
    total_tests = len(TEST_CASES) + len(EXTENDED_TEST_CASES)

    print(f"\nRunning Comprehensive Test Suite: Backend vs Middleware ({total_tests} tests)\n")

    # Combine regular and extended test cases
    all_cases = TEST_CASES + EXTENDED_TEST_CASES

    for tc in all_cases:
        name = tc["name"]
        expr = tc["expr"]
        raw_interval = tc["interval"]
        integrate_n = tc.get("integrate_n")       # Phase 3: explicit n
        use_mpmath = tc.get("use_mpmath", False)   # Phase 3: high-precision mode

        # --- 1. DIRECT BACKEND (Ground Truth) ---
        dir_val, dir_fam, dir_conv, dir_nnodes = None, "ERR", False, 0
        try:
            clean_interval = raw_interval
            if clean_interval is not None:
                clean_interval = tuple(
                    float("inf") if str(x) in ("Infinity", "inf") else
                    -float("inf") if str(x) in ("-Infinity", "-inf") else x
                    for x in raw_interval
                )

            analysis = analyzer.analyze(expr, interval=clean_interval)
            dir_fam = analysis.recommended_family.value

            kw = {"expression": expr, "interval": clean_interval}
            if integrate_n is not None:
                kw["n"] = integrate_n
            if use_mpmath:
                kw["use_mpmath"] = True

            integration = analyzer.execute_quadrature(**kw)
            dir_val = integration.value
            dir_conv = integration.converged
            dir_nnodes = integration.n_nodes
        except Exception as e:
            dir_val = f"Err: {str(e)[:20]}"

        # --- 2. HTTP MIDDLEWARE (Proxy under test) ---
        api_val, api_fam, api_conv, api_err = None, "ERR", False, None
        api_interval = _format_api_interval(raw_interval)

        try:
            # Analyze via API
            analyze_payload = {"expression": expr, "interval": api_interval}
            res_a = requests.post(f"{API_URL}/analyze", json=analyze_payload, timeout=30)
            if res_a.status_code == 200:
                api_fam = res_a.json().get("recommended_family", "Unknown")
            else:
                # Phase 1: capture error detail
                try:
                    detail = res_a.json().get("detail", "")[:30]
                except Exception:
                    detail = ""
                api_fam = f"HTTP {res_a.status_code}{': ' + detail if detail else ''}"

            # Integrate via API
            integrate_payload = {"expression": expr, "interval": api_interval}
            if integrate_n is not None:
                integrate_payload["n"] = integrate_n
            if use_mpmath:
                integrate_payload["use_mpmath"] = True

            res_i = requests.post(f"{API_URL}/integrate", json=integrate_payload, timeout=30)
            if res_i.status_code == 200:
                api_data = res_i.json()
                api_val = api_data.get("value")
                api_conv = api_data.get("converged", False)
                api_err = api_data.get("error_estimate")
            else:
                # Phase 1: capture error detail
                try:
                    detail = res_i.json().get("detail", "")[:30]
                except Exception:
                    detail = ""
                api_val = f"HTTP {res_i.status_code}{': ' + detail if detail else ''}"
        except requests.exceptions.ConnectionError:
            api_val = "NoConn"
        except requests.exceptions.Timeout:
            api_val = "Timeout"
        except Exception as e:
            api_val = f"Err{str(e)[:15]}"

        # --- 3. COMPARE FIDELITY (Phase 2: expanded field comparison) ---
        match = True

        # Both sides errored → fidelity OK (middleware correctly propagates errors)
        both_errored = isinstance(dir_val, str) and dir_val.startswith("Err:") \
                       and isinstance(api_val, str) and api_val.startswith("HTTP")

        if not both_errored:
            # Family match
            if dir_fam != api_fam:
                match = False

            # Value match (robust comparator)
            if not values_match(dir_val, api_val):
                match = False

            # Convergence match (Phase 2)
            if dir_conv != api_conv:
                match = False

        match_str = "✅ YES" if match else "❌ NO"
        if match:
            passed_fidelity += 1

        fam_str = f"{dir_fam[:5]}/{api_fam[:5]}"
        conv_str = f"{'Y' if dir_conv else 'N'}/{'Y' if api_conv else 'N'}"

        dir_val_str = f"{dir_val:.6e}" if isinstance(dir_val, float) and math.isfinite(dir_val) else str(dir_val)
        api_val_str = f"{api_val:.6e}" if isinstance(api_val, float) and math.isfinite(api_val) else str(api_val)
        err_str     = f"{api_err:.2e}"  if isinstance(api_err, float) and math.isfinite(api_err) else str(api_err)

        table.add_row([name[:18], fam_str, conv_str, dir_val_str, api_val_str, err_str, match_str])

    print(table)
    print(f"\nFidelity Score: {passed_fidelity}/{total_tests} tests passed 1-to-1 mapping.")

    if passed_fidelity == total_tests:
        print("🎉 Middleware flawlessly proxies requests to the backend!")
    else:
        print("⚠️  Middleware introduced variations. Check logs for NaN/Inf serialization issues.")

    # --- Error-handling tests (Phase 3) ---
    if ERROR_TEST_CASES:
        print(f"\n--- Error Handling Tests ({len(ERROR_TEST_CASES)} cases) ---\n")
        err_passed = 0
        for tc in ERROR_TEST_CASES:
            name = tc["name"]
            expr = tc["expr"]
            api_interval = _format_api_interval(tc.get("interval"))

            # Direct backend error
            dir_err = False
            try:
                analyzer.analyze(expr, interval=tc.get("interval"))
                analyzer.execute_quadrature(expr, interval=tc.get("interval"))
            except Exception:
                dir_err = True

            # API error
            api_err_resp = False
            try:
                res = requests.post(
                    f"{API_URL}/integrate",
                    json={"expression": expr, "interval": api_interval},
                    timeout=5
                )
                if res.status_code != 200:
                    api_err_resp = True
            except Exception:
                api_err_resp = True

            ok = dir_err and api_err_resp
            status = "✅" if ok else "❌"
            if ok:
                err_passed += 1
            print(f"  {status} {name}: Direct={'Err' if dir_err else 'OK'} / API={'Err' if api_err_resp else 'OK'}")

        print(f"\nError Handling: {err_passed}/{len(ERROR_TEST_CASES)} passed.")


if __name__ == "__main__":
    run_comparison()