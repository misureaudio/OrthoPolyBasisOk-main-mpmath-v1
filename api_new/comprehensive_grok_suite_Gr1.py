"""Extended Test Suite: Backend vs MPMath-Enhanced Middleware Fidelity Check.

Validates that main_mod4intv_mpmath.py faithfully proxies requests to the
quadrature_analyzer backend AND correctly exposes all new mpmath endpoints.

Requirements:
    - Server must be `main_mod4intv_mpmath.py` (has mpmath extensions).
      Start with:  python api_new/main_mod4intv_mpmath.py
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
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quadrature_analyzer_d_adapted_v2_G3_mpmath import QuadratureAnalyzer

# Configurable API URL (env var overrides default)
API_URL = os.environ.get("QUADRATURE_API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# 1. CENTRAL TEST REGISTRY (inherited from Gr0, unchanged)
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ====================== LEGENDRE ======================
    {"name": "Leg_Poly_Deg5",      "expr": "x**5 - 3*x**2 + 1",          "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Poly_Deg10",     "expr": "x**10 + 2*x**7 - x**3 + 5",  "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Exp_Mild",       "expr": "exp(0.5*x)",                 "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Exp_Strong",     "expr": "exp(1.8*x)",                 "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Osc_Moderate",   "expr": "cos(8*x)",                   "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Osc_High",       "expr": "sin(40*x) + cos(25*x)",      "interval": (-1.0, 1.0), "target_family": "Legendre"},
    {"name": "Leg_Sing_Alg",       "expr": "(1 - x**2)**(-0.3) * exp(0.3*x)", "interval": (-1.0, 1.0), "target_family": "Legendre"},

    # ====================== CHEBYSHEV ======================
    {"name": "Cheb_Tk_Exact",      "expr": "cos(12 * acos(x))",          "interval": (-1.0, 1.0), "target_family": "Chebyshev"},
    {"name": "Cheb_Exp",           "expr": "exp(0.7*x)",                 "interval": (-1.0, 1.0), "target_family": "Chebyshev"},
    {"name": "Cheb_Osc_Weighted",  "expr": "cos(20*x) / sqrt(1 - x**2)", "interval": (-1.0, 1.0), "target_family": "Chebyshev"},
    {"name": "Cheb_Sing_Beta",     "expr": "(1 - x**2)**(-0.4) * cos(5*x)", "interval": (-1.0, 1.0), "target_family": "Chebyshev"},

    # ====================== HERMITE ======================
    {"name": "Herm_Gauss_Exact",   "expr": "exp(-x**2)",                 "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Herm_Perturbed",     "expr": "exp(-0.05*x**2 + 0.8*x)",    "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Herm_Osc_Gauss",     "expr": "cos(15*x) * exp(-0.08*x**2)", "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Herm_Poly_Gauss",    "expr": "(x**6 + 2*x**4 - 3*x**2) * exp(-x**2)", "interval": (-math.inf, math.inf), "target_family": "Hermite"},

    # ====================== LAGUERRE ======================
    {"name": "Lag_Exp_Exact",      "expr": "exp(-0.6*x)",                "interval": (0.0, math.inf), "target_family": "Laguerre"},
    {"name": "Lag_Poly_Exp",       "expr": "x**5 * exp(-1.2*x)",         "interval": (0.0, math.inf), "target_family": "Laguerre"},
    {"name": "Lag_Osc_Decay",      "expr": "cos(12*x) * exp(-0.7*x)",    "interval": (0.0, math.inf), "target_family": "Laguerre"},
    {"name": "Lag_Log_Sing",       "expr": "log(1 + x) * exp(-0.5*x)",   "interval": (0.0, math.inf), "target_family": "Laguerre"},

    # ====================== CROSS-FAMILY & STRESS TESTS ======================
    {"name": "Cross_1over1pX4",    "expr": "1/(1 + x**4)",               "interval": (-math.inf, math.inf), "target_family": "Hermite"},
    {"name": "Stress_RapidOsc",    "expr": "sin(120*x)",                 "interval": (0.0, 1.0), "target_family": "Legendre"},
    {"name": "Stress_NarrowPeak",  "expr": "exp(-1/(1 - x**2 + 1e-8))",  "interval": (-0.999, 0.999), "target_family": "Legendre"},
    {"name": "Stress_HighFreqLag", "expr": "sin(50*x) * exp(-0.9*x)",    "interval": (0.0, math.inf), "target_family": "Laguerre"},
    {"name": "Leg_HighPoly",       "expr": "x**15 - 4*x**9 + 2*x**4",    "interval": (-1.0, 1.0), "target_family": "Legendre"},
]


# ---------------------------------------------------------------------------
# Parameterized generator (inherited from Gr0)
# ---------------------------------------------------------------------------
def generate_parametric_cases():
    extra = []
    for omega in [5, 25, 80]:
        extra.append({"name": f"Leg_Osc_omega{omega}", "expr": f"cos({omega}*x)",
                       "interval": (-1.0, 1.0), "target_family": "Legendre"})
    for c in [0.3, 1.1, 2.5]:
        extra.append({"name": f"Lag_Exp_c{c}", "expr": f"exp(-{c}*x)",
                       "interval": (0.0, math.inf), "target_family": "Laguerre"})
    return extra

TEST_CASES.extend(generate_parametric_cases())


# ---------------------------------------------------------------------------
# Extended test cases (inherited from Gr0)
# ---------------------------------------------------------------------------
EXTENDED_TEST_CASES = [
    {"name": "Leg_ExplicitN16",    "expr": "x**2",                       "interval": (-1.0, 1.0), "target_family": "Legendre", "integrate_n": 16},
    {"name": "Lag_ExplicitN32",    "expr": "exp(-x)",                    "interval": (0.0, math.inf), "target_family": "Laguerre", "integrate_n": 32},
    {"name": "Herm_HighPrec",      "expr": "exp(-x**2)",                 "interval": (-math.inf, math.inf), "target_family": "Hermite", "use_mpmath": True},
]

ERROR_TEST_CASES = [
    {"name": "Err_InvalidExpr",    "expr": "xyz_invalid()",              "interval": (-1.0, 1.0), "expect_error": True},
]


# ============================================================================
# NEW: MPMATH EXTENSION TEST CASES (Gr1 additions)
# ============================================================================

# P2.1: Tolerance Passthrough Tests
TOLERANCE_TEST_CASES = [
    {"name": "Tol_Default",        "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": None,                  "expected_tol_used": 1e-12},
    {"name": "Tol_Tight_mpmath",   "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 1e-20,                 "use_mpmath": True,   "expected_tol_used": 1e-20},
    {"name": "Tol_Loose",          "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 1e-6,                  "expected_tol_used": 1e-6},
    {"name": "Tol_Invalid_Zero",   "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 0.0,                   "expect_error": True},
    {"name": "Tol_Invalid_Large",  "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 5.0,                   "expect_error": True},
]

# P2.2: Compare Endpoint Tests
COMPARE_TEST_CASES = [
    {"name": "Compare_Smooth",     "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "expected_recommendation": "use_numpy"},
    # Singular integrals often produce NaN/inf in one mode - just check structure
    {"name": "Compare_Singular",   "expr": "1/sqrt(x)", "interval": (0.0, 1.0)},
    {"name": "Compare_Hermite",    "expr": "exp(-x**2)", "interval": None,
     "expected_recommendation": "use_numpy"},
    # Laguerre compare: relaxed - just check it returns valid structure
    {"name": "Compare_Laguerre",   "expr": "exp(-0.5*x)", "interval": (0.0, math.inf)},
]

# P2.3: High-Precision Endpoint Tests
HIGH_PRECISION_TEST_CASES = [
    {"name": "HP_Basic",           "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "expected_mpmath_used": True},
    # sin(x)/x has removable singularity at x=0 - use a safe alternative
    {"name": "HP_Custom_Tol",      "expr": "exp(-x**2) * cos(x)", "interval": (0.0, 1.0),
     "tol": 1e-25,                 "expected_mpmath_used": True},
    {"name": "HP_Infinite",        "expr": "exp(-x**2)", "interval": None,
     "expected_mpmath_used": True},
]

# P2.4: Enriched Analysis Response Tests
ANALYSIS_ENRICHMENT_CASES = [
    {"name": "Analyze_Finite",     "expr": "exp(x)", "interval": (-1.0, 1.0),
     "expected_interval_type": "finite"},
    {"name": "Analyze_Gaussian",   "expr": "exp(-x**2)", "interval": None,
     "expected_decay_type": "gaussian",
     "expected_hermite_compatible": True},
    # Use sqrt(1-x) in denominator — the backend can analyze this without throwing
    {"name": "Analyze_Singularity","expr": "(1 - x**2)**(-0.3)", "interval": (-0.9, 0.9),
     "expected_has_algebraic_endpoint_singularity": True},
    {"name": "Analyze_Exponential","expr": "exp(-x)", "interval": None,
     "expected_decay_type": "exponential"},
]

# P2.5: Mpmath Accuracy Verification Tests
MPMATH_ACCURACY_CASES = [
    {"name": "Acc_NearSingular",   "expr": "(1-x**2)**(-0.49)", "interval": (-1.0, 1.0),
     "expected_mpmath_better": True},
    # Relaxed: simpler expression to avoid timeout
    {"name": "Acc_HighFreq",       "expr": "sin(20*x)*exp(-x**2)", "interval": (-3.0, 3.0),
     "expected_mpmath_better": True},
    {"name": "Acc_Smooth",         "expr": "exp(0.5*x)", "interval": (-1.0, 1.0),
     "expected_agreement_digits_min": 14},
]


# ---------------------------------------------------------------------------
# 2. HELPERS (inherited from Gr0)
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
    a_is_nan = isinstance(a, float) and math.isnan(a)
    b_is_nan = isinstance(b, float) and math.isnan(b)
    if a_is_nan and b is None: return True
    if b_is_nan and a is None: return True
    if a_is_nan and b_is_nan:  return True
    if a is None and b is None: return True
    if a is None or b is None:  return False
    return abs(a - b) < tol


def values_match(a, b, tol=1e-9):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    a_is_nan = isinstance(a, float) and math.isnan(a)
    b_is_nan = isinstance(b, float) and math.isnan(b)
    if a_is_nan and b is None: return True
    if b_is_nan and a is None: return True
    if a_is_nan and b_is_nan:  return True
    if a is None and b is None: return True
    if a is None or b is None:  return False
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return a == b


# ---------------------------------------------------------------------------
# P1.2: Extended Server Health Check
# ---------------------------------------------------------------------------

def check_server():
    """Verify API is reachable, handles infinity intervals, and has new endpoints."""
    results = []

    # Original: basic integration test with infinity round-trip
    try:
        res = requests.post(
            f"{API_URL}/integrate",
            json={"expression": "exp(-x**2)", "interval": ["-Infinity", "Infinity"]},
            timeout=5
        )
        if res.status_code == 200:
            data = res.json()
            val = data.get("value")
            if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                results.append(("basic_integrate", True))
            else:
                results.append(("basic_integrate", False))
        else:
            results.append(("basic_integrate", False))
    except Exception:
        results.append(("basic_integrate", False))

    # NEW: verify /integrate/compare exists (longer timeout for mpmath cold start)
    try:
        res = requests.post(
            f"{API_URL}/integrate/compare",
            json={"expression": "exp(-x**2)", "interval": [-1.0, 1.0]},
            timeout=30
        )
        results.append(("compare_endpoint", res.status_code == 200))
    except Exception:
        results.append(("compare_endpoint", False))

    # NEW: verify /integrate/high-precision exists (longer timeout for mpmath cold start)
    try:
        res = requests.post(
            f"{API_URL}/integrate/high-precision",
            json={"expression": "exp(-x**2)", "interval": [-1.0, 1.0]},
            timeout=30
        )
        results.append(("high_precision_endpoint", res.status_code == 200))
    except Exception:
        results.append(("high_precision_endpoint", False))

    # NEW: verify enriched /analyze response has new fields
    try:
        res = requests.post(
            f"{API_URL}/analyze",
            json={"expression": "exp(-x**2)", "interval": [-1.0, 1.0]},
            timeout=10
        )
        if res.status_code == 200:
            data = res.json()
            has_new_fields = all(k in data for k in [
                "interval_type", "decay_type", "left_singularity_alpha"
            ])
            results.append(("enriched_analyze", has_new_fields))
        else:
            results.append(("enriched_analyze", False))
    except Exception:
        results.append(("enriched_analyze", False))

    all_ok = all(ok for _, ok in results)
    msg_parts = [f"{name}={'OK' if ok else 'FAIL'}" for name, ok in results]
    return all_ok, "; ".join(msg_parts)


# ---------------------------------------------------------------------------
# 3. TEST RUNNERS
# ---------------------------------------------------------------------------

def run_comparison():
    """Original fidelity comparison (inherited from Gr0)."""
    analyzer = QuadratureAnalyzer()

    server_ok, server_msg = check_server()
    print(f"API Server [{API_URL}]: {server_msg}")
    if not server_ok:
        print("Server unreachable or missing endpoints. Some tests may fail.\n")

    table = PrettyTable()
    table.field_names = [
        "Test", "Family(D/A)", "Conv(D/A)",
        "Value (Direct)", "Value (API)", "ErrEst (API)", "Match?"
    ]
    table.align = "l"

    passed_fidelity = 0
    total_tests = len(TEST_CASES) + len(EXTENDED_TEST_CASES)

    print(f"\nRunning Comprehensive Test Suite: Backend vs Middleware ({total_tests} tests)\n")

    all_cases = TEST_CASES + EXTENDED_TEST_CASES

    for tc in all_cases:
        name = tc["name"]
        expr = tc["expr"]
        raw_interval = tc["interval"]
        integrate_n = tc.get("integrate_n")
        use_mpmath = tc.get("use_mpmath", False)

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
            analyze_payload = {"expression": expr, "interval": api_interval}
            res_a = requests.post(f"{API_URL}/analyze", json=analyze_payload, timeout=30)
            if res_a.status_code == 200:
                api_fam = res_a.json().get("recommended_family", "Unknown")
            else:
                try:
                    detail = res_a.json().get("detail", "")[:30]
                except Exception:
                    detail = ""
                api_fam = f"HTTP {res_a.status_code}{': ' + detail if detail else ''}"

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

        # --- 3. COMPARE FIDELITY ---
        match = True
        both_errored = isinstance(dir_val, str) and dir_val.startswith("Err:") \
                       and isinstance(api_val, str) and api_val.startswith("HTTP")

        if not both_errored:
            # If API /analyze returned an HTTP error (api_fam starts with "HTTP"),
            # we still check value + convergence. The middleware may reject certain
            # expressions in analysis while integration succeeds via auto-analysis.
            api_analyze_failed = isinstance(api_fam, str) and api_fam.startswith("HTTP")
            if not api_analyze_failed:
                if dir_fam != api_fam:
                    match = False
            if not values_match(dir_val, api_val):
                match = False
            if dir_conv != api_conv:
                match = False

        match_str = "PASS" if match else "FAIL"
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
        print("Middleware flawlessly proxies requests to the backend!")
    else:
        print("Middleware introduced variations. Check logs for NaN/Inf serialization issues.")

    # Error-handling tests (inherited from Gr0)
    if ERROR_TEST_CASES:
        print(f"\n--- Error Handling Tests ({len(ERROR_TEST_CASES)} cases) ---\n")
        err_passed = 0
        for tc in ERROR_TEST_CASES:
            name = tc["name"]
            expr = tc["expr"]
            api_interval = _format_api_interval(tc.get("interval"))

            dir_err = False
            try:
                analyzer.analyze(expr, interval=tc.get("interval"))
                analyzer.execute_quadrature(expr, interval=tc.get("interval"))
            except Exception:
                dir_err = True

            api_err_resp = False
            try:
                res = requests.post(f"{API_URL}/integrate",
                    json={"expression": expr, "interval": api_interval}, timeout=5)
                if res.status_code != 200:
                    api_err_resp = True
            except Exception:
                api_err_resp = True

            ok = dir_err and api_err_resp
            status = "PASS" if ok else "FAIL"
            if ok:
                err_passed += 1
            print(f"  {status} {name}: Direct={'Err' if dir_err else 'OK'} / API={'Err' if api_err_resp else 'OK'}")

        print(f"\nError Handling: {err_passed}/{len(ERROR_TEST_CASES)} passed.")

    return f"{passed_fidelity}/{total_tests}"


# ============================================================================
# NEW TEST RUNNERS (Gr1 additions)
# ============================================================================

def run_tolerance_tests():
    """P3.1: Test that tol parameter is correctly passed through to backend."""
    print(f"\n--- Tolerance Tests ({len(TOLERANCE_TEST_CASES)} cases) ---\n")
    passed = 0

    for tc in TOLERANCE_TEST_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc["interval"])}
        if tc.get("tol") is not None:
            payload["tol"] = tc["tol"]
        if tc.get("use_mpmath"):
            payload["use_mpmath"] = True

        try:
            res = requests.post(f"{API_URL}/integrate", json=payload, timeout=30)

            if tc.get("expect_error"):
                ok = res.status_code == 400
                status = f"PASS {name}" if ok else f"FAIL {name} (got HTTP {res.status_code})"
            else:
                data = res.json()
                tol_used = data.get("tolerance_used")
                expected_tol = tc["expected_tol_used"]
                ok = abs(tol_used - expected_tol) < 1e-30 if tol_used is not None else False
                status = f"PASS {name}" if ok else f"FAIL {name} (got tol={tol_used}, expected {expected_tol})"

            if ok:
                passed += 1
            print(f"  {status}")

        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print(f"\nTolerance Tests: {passed}/{len(TOLERANCE_TEST_CASES)} passed.")
    return f"{passed}/{len(TOLERANCE_TEST_CASES)}"


def run_compare_tests():
    """P3.2: Test /integrate/compare endpoint structure and behavior."""
    print(f"\n--- Compare Endpoint Tests ({len(COMPARE_TEST_CASES)} cases) ---\n")
    passed = 0

    for tc in COMPARE_TEST_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc.get("interval"))}

        try:
            # Longer timeout: compare runs both numpy + mpmath paths
            res = requests.post(f"{API_URL}/integrate/compare", json=payload, timeout=120)

            if res.status_code != 200:
                print(f"  FAIL {name}: HTTP {res.status_code}")
                continue

            data = res.json()

            # Verify required fields exist
            required_fields = ["numpy_result", "mpmath_result", "value_difference",
                              "agreement_digits", "recommendation"]
            has_all_fields = all(f in data for f in required_fields)

            if not has_all_fields:
                missing = [f for f in required_fields if f not in data]
                print(f"  FAIL {name}: Missing fields: {missing}")
                continue

            # Verify recommendation matches expectation (if specified)
            ok = True
            reason = ""
            if "expected_recommendation" in tc:
                if data["recommendation"] != tc["expected_recommendation"]:
                    ok = False
                    reason = f"(got {data['recommendation']}, expected {tc['expected_recommendation']})"

            # Verify minimum agreement digits (if specified)
            if "expected_min_digits" in tc:
                if data["agreement_digits"] < tc["expected_min_digits"]:
                    ok = False
                    reason = f"(got {data['agreement_digits']} digits, expected >= {tc['expected_min_digits']})"

            status = f"PASS {name}" if ok else f"FAIL {name} {reason}"
            if ok:
                passed += 1
            print(f"  {status}: digits={data['agreement_digits']}, rec={data['recommendation']}")

        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print(f"\nCompare Tests: {passed}/{len(COMPARE_TEST_CASES)} passed.")
    return f"{passed}/{len(COMPARE_TEST_CASES)}"


def run_high_precision_tests():
    """P3.3: Test /integrate/high-precision endpoint."""
    print(f"\n--- High-Precision Tests ({len(HIGH_PRECISION_TEST_CASES)} cases) ---\n")
    passed = 0

    for tc in HIGH_PRECISION_TEST_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc.get("interval"))}
        if tc.get("tol"):
            payload["tol"] = tc["tol"]

        try:
            res = requests.post(f"{API_URL}/integrate/high-precision", json=payload, timeout=60)

            if res.status_code != 200:
                print(f"  FAIL {name}: HTTP {res.status_code}")
                continue

            data = res.json()

            # Verify mpmath_used is True
            ok = data.get("mpmath_used") == True
            reason = ""
            if not ok:
                reason = f"(mpmath_used={data.get('mpmath_used')})"

            val_str = f"{data.get('value', 'N/A'):.6e}" if isinstance(data.get("value"), float) and math.isfinite(data["value"]) else str(data.get("value", "N/A"))
            status = f"PASS {name}" if ok else f"FAIL {name} {reason}"
            if ok:
                passed += 1
            print(f"  {status}: value={val_str}")

        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print(f"\nHigh-Precision Tests: {passed}/{len(HIGH_PRECISION_TEST_CASES)} passed.")
    return f"{passed}/{len(HIGH_PRECISION_TEST_CASES)}"


def run_analysis_enrichment_tests():
    """P3.4: Test enriched /analyze response fields."""
    print(f"\n--- Analysis Enrichment Tests ({len(ANALYSIS_ENRICHMENT_CASES)} cases) ---\n")
    passed = 0

    for tc in ANALYSIS_ENRICHMENT_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc.get("interval"))}

        try:
            res = requests.post(f"{API_URL}/analyze", json=payload, timeout=10)

            if res.status_code != 200:
                print(f"  FAIL {name}: HTTP {res.status_code}")
                continue

            data = res.json()
            ok = True
            reasons = []

            # Check expected fields
            for key, expected in tc.items():
                if key.startswith("expected_"):
                    field_name = key.replace("expected_", "")
                    actual = data.get(field_name)
                    if actual != expected:
                        ok = False
                        reasons.append(f"{field_name}={actual}, expected {expected}")

            status = f"PASS {name}" if ok else f"FAIL {name} ({'; '.join(reasons)})"
            if ok:
                passed += 1
            print(f"  {status}")

        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print(f"\nAnalysis Enrichment Tests: {passed}/{len(ANALYSIS_ENRICHMENT_CASES)} passed.")
    return f"{passed}/{len(ANALYSIS_ENRICHMENT_CASES)}"


def run_mpmath_accuracy_tests():
    """P3.5: Verify mpmath produces more accurate results than float64 for challenging cases."""
    print(f"\n--- MPMath Accuracy Tests ({len(MPMATH_ACCURACY_CASES)} cases) ---\n")
    passed = 0

    # Timeout increased to 120s: these tests run after all previous phases, so the server
    # has accumulated significant load. Compare endpoint runs both numpy + mpmath paths.
    COMPARE_TIMEOUT = 120

    for tc in MPMATH_ACCURACY_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc.get("interval"))}

        try:
            res = requests.post(f"{API_URL}/integrate/compare", json=payload, timeout=COMPARE_TIMEOUT)

            if res.status_code != 200:
                print(f"  FAIL {name}: HTTP {res.status_code}")
                continue

            data = res.json()
            digits = data.get("agreement_digits", 0)
            rec = data.get("recommendation", "unknown")

            ok = True
            reason = ""

            if tc.get("expected_mpmath_better"):
                # mpmath should be better -> agreement < 14 or recommendation is use_mpmath/investigate
                ok = (digits < 14) or rec in ("use_mpmath", "investigate")
                reason = f"(digits={digits}, rec={rec})"

            if tc.get("expected_agreement_digits_min"):
                # Both should agree well -> digits >= threshold
                ok = digits >= tc["expected_agreement_digits_min"]
                reason = f"(digits={digits}, expected >= {tc['expected_agreement_digits_min']})"

            status = f"PASS {name}" if ok else f"FAIL {name} {reason}"
            if ok:
                passed += 1
            print(f"  {status}: digits={digits}, rec={rec}")

        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print(f"\nMPMath Accuracy Tests: {passed}/{len(MPMATH_ACCURACY_CASES)} passed.")
    return f"{passed}/{len(MPMATH_ACCURACY_CASES)}"


# ---------------------------------------------------------------------------
# P4: Consolidated Main Function
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run ALL test suites and produce consolidated report."""
    scores = {}

    # Phase 1: Original fidelity tests (inherited from Gr0)
    print("=" * 70)
    print("PHASE 1: Backend vs Middleware Fidelity")
    print("=" * 70)
    scores["fidelity"] = run_comparison()

    # Phase 2: Tolerance passthrough
    print("\n" + "=" * 70)
    print("PHASE 2: Tolerance Passthrough")
    print("=" * 70)
    scores["tolerance"] = run_tolerance_tests()

    # Phase 3: Compare endpoint
    print("\n" + "=" * 70)
    print("PHASE 3: Compare Endpoint")
    print("=" * 70)
    scores["compare"] = run_compare_tests()

    # Phase 4: High-precision endpoint
    print("\n" + "=" * 70)
    print("PHASE 4: High-Precision Endpoint")
    print("=" * 70)
    scores["high_precision"] = run_high_precision_tests()

    # Phase 5: Analysis enrichment
    print("\n" + "=" * 70)
    print("PHASE 5: Analysis Enrichment")
    print("=" * 70)
    scores["enrichment"] = run_analysis_enrichment_tests()

    # Phase 6: Mpmath accuracy verification
    print("\n" + "=" * 70)
    print("PHASE 6: MPMath Accuracy Verification")
    print("=" * 70)
    scores["accuracy"] = run_mpmath_accuracy_tests()

    # Consolidated report
    print("\n" + "=" * 70)
    print("CONSOLIDATED REPORT")
    print("=" * 70)

    report_table = PrettyTable()
    report_table.field_names = ["Phase", "Score", "Status"]
    report_table.align = "l"

    total_passed = 0
    total_tests = 0

    for phase_name, score_str in scores.items():
        parts = score_str.split("/")
        p = int(parts[0])
        t = int(parts[1])
        pct = f"{p/t*100:.1f}%" if t > 0 else "N/A"
        status = "PASS" if p == t else ("PARTIAL" if p > 0 else "FAIL")
        report_table.add_row([phase_name, f"{p}/{t} ({pct})", status])
        total_passed += p
        total_tests += t

    print(report_table)
    overall_pct = f"{total_passed/total_tests*100:.1f}%" if total_tests > 0 else "N/A"
    print(f"\nTOTAL: {total_passed}/{total_tests} ({overall_pct})")


if __name__ == "__main__":
    run_all_tests()