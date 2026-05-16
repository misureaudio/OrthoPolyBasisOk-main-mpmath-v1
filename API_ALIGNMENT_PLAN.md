# Alignment Plan: `comprehensive_grok_suite_Gr0.py` → Existing API

**Date:** 2026-05-12  
**Status:** DRAFT — for review  
**Constraint:** NO modifications to `legendre/`, `chebyshev/`, `hermite/`, `laguerre/` subdirectories NOR `api/quadrature_analyzer.py`

---

## 1. Executive Summary

`comprehensive_grok_suite_Gr0.py` is a test/comparison suite that validates the FastAPI middleware (`main.py` / `main_mod4intv.py`) faithfully proxies requests to the backend (`quadrature_analyzer.py`). This plan identifies all misalignments between the test suite and the current API contract, then prescribes targeted changes **only** to `comprehensive_grok_suite_Gr0.py`.

---

## 2. Current Architecture (Reference)

```
┌─────────────────────────────────────────────────────────────┐
│  comprehensive_grok_suite_Gr0.py                            │
│  ── Path A: Direct Backend (Ground Truth)                   │
│     QuadratureAnalyzer().analyze() / execute_quadrature()   │
│  ── Path B: HTTP Middleware (Proxy under test)              │
│     POST http://localhost:8000/analyze                      │
│     POST http://localhost:8000/integrate                    │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼─────────────────────────────────────────────────┐
    │  main.py / main_mod4intv.py (FastAPI server)            │
    │                                                          │
    │  POST /analyze  → AnalysisRequest                       │
    │                  {expression, interval?, variable,       │
    │                   max_deriv_order}                      │
    │                  → FunctionAnalysis response             │
    │                                                          │
    │  POST /integrate → IntegrationRequest                   │
    │                  {expression, interval?, n?, variable,   │
    │                   use_mpmath}                           │
    │                  → QuadratureResult response             │
    └──────────┬─────────────────────────────────────────────┘
               │
    ┌──────────▼─────────────────────────────────────────────┐
    │  quadrature_analyzer.py (NEVER MODIFY)                  │
    │                                                          │
    │  QuadratureAnalyzer.analyze() → FunctionAnalysis        │
    │  QuadratureAnalyzer.execute_quadrature() →              │
    │      QuadratureResult                                   │
    └──────────┬─────────────────────────────────────────────┘
               │
    ┌──────────▼─────────────────────────────────────────────┐
    │  legendre/ chebyshev/ hermite/ laguerre/ (NEVER MODIFY) │
    │                                                          │
    │  LegendreQuadrature(n).integrate(f)                     │
    │  clencurt_integrate_interval(f, a, b, n)                │
    │  GaussHermiteQuadrature(n).integrate(f)                 │
    │  LaguerreQuadrature(n, alpha).integrate(f)              │
    └─────────────────────────────────────────────────────────┘
```

---

## 3. Identified Misalignments

### 3.1 Import Path Inconsistency (CRITICAL)

| Aspect | Current `comprehensive_grok_suite_Gr0.py` | Correct (matches `main.py`) |
|--------|-------------------------------------------|------------------------------|
| sys.path | `os.path.dirname(os.path.abspath(__file__))` → adds `api/` | `os.path.dirname(os.path.dirname(...))` → adds project root |
| import | `from quadrature_analyzer import QuadratureAnalyzer` | Same, but resolved from **project root** |

**Problem:** The test suite resolves `quadrature_analyzer` from the `api/` directory directly. If a different copy exists at the project root level, or if module resolution differs between the test runner and the server, ground-truth results will diverge from API results for reasons unrelated to middleware fidelity.

**Fix:** Align sys.path to match `main.py`:
```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

---

### 3.2 Server Target Mismatch (MEDIUM)

| Aspect | Current | Recommended |
|--------|---------|-------------|
| API URL | Hardcoded `"http://localhost:8000"` | Same, but the test suite does not specify which server variant it targets |

**Problem:** There are TWO server variants:
- `main.py` — basic, no interval middleware
- `main_mod4intv.py` — has `_sanitize_interval()` middleware that converts string `"Infinity"` → `float("inf")`

The test suite's `_format_api_interval()` converts `float("inf")` → `"Infinity"` for JSON payloads. This **requires** the `main_mod4intv.py` middleware to work correctly, because standard FastAPI/Pydantic will reject or mishandle string infinity values in a `Tuple[float, float]` field.

**Fix:** 
- Make the target server configurable (CLI arg or env var)
- Document that `main_mod4intv.py` is the required server for full test coverage
- Add a startup health-check that verifies interval handling works

---

### 3.3 Interval Formatting — String vs Numeric Infinity (HIGH)

| Component | Behavior |
|-----------|----------|
| `_format_api_interval()` in test suite | Converts `float("inf")` → `"Infinity"` string |
| `main_mod4intv.py` middleware | Converts `"Infinity"` string → `float("inf")` |
| `main.py` (no middleware) | Would receive `"Infinity"` string into `Tuple[float, float]` — **Pydantic coercion behavior undefined** |

**Problem:** The round-trip `float("inf")` → `"Infinity"` → `float("inf")` depends entirely on the middleware in `main_mod4intv.py`. If tested against `main.py`, infinite-interval tests will fail for reasons unrelated to fidelity.

**Fix:** Add a pre-flight test that sends an infinite interval and verifies the server round-trips it correctly. Skip or flag infinite-interval tests if the server does not support string infinity.

---

### 3.4 Unused API Parameters (LOW)

The existing API supports parameters the test suite never exercises:

| Endpoint | Parameter | Purpose | Test Suite Usage |
|----------|-----------|---------|------------------|
| `/analyze` | `variable` | Independent variable name (default `"x"`) | Never set — always defaults to `"x"` |
| `/analyze` | `max_deriv_order` | Derivative probing depth (default 6) | Never set — always defaults to 6 |
| `/integrate` | `variable` | Independent variable name | Never set |
| `/integrate` | `use_mpmath` | Arbitrary-precision arithmetic | Never set — always defaults to `False` |

**Fix:** Add a small set of test cases that exercise non-default parameters:
- A case with `variable="t"` 
- A case with `use_mpmath=True` for high-precision comparison

---

### 3.5 Incomplete Response Field Validation (MEDIUM)

| Response Source | Fields Available | Fields Validated by Test Suite |
|-----------------|------------------|-------------------------------|
| `/analyze` | `original_expr`, `recommended_family`, `confidence`, `recommendation_reason`, `suggested_min_n`, `suggested_max_n`, `interval_a`, `interval_b`, `singularities`, `is_periodic`, `derivative_growth` | Only `recommended_family` |
| `/integrate` | `value`, `family_used`, `n_nodes`, `converged`, `error_estimate`, `message` | Only `value` |

**Problem:** The test suite compares only 2 of ~17 available response fields. Middleware bugs in other fields (e.g., `_safe_float` converting important values to `None`) would go undetected.

**Fix:** Expand comparison to validate:
- From `/analyze`: `suggested_min_n`, `suggested_max_n`, `confidence`, `is_periodic`, `derivative_growth`
- From `/integrate`: `family_used`, `n_nodes`, `converged`, `error_estimate`

---

### 3.6 Tolerance Mismatch (LOW)

| Component | Tolerance |
|-----------|-----------|
| `float_eq()` / `values_match()` in test suite | `tol=1e-9` |
| `QuadratureAnalyzer.execute_quadrature()` convergence criterion | `tol=1e-12` |

**Problem:** The quadrature engine converges to ~1e-12, but the test suite accepts 1e-9. This is actually *safer* (more lenient), so it is not a bug — but documenting the discrepancy prevents confusion.

**Fix:** No code change needed. Document that `tol=1e-9` in the test suite is intentionally more lenient than the engine's internal `tol=1e-12`.

---

### 3.7 Error Handling Gaps (MEDIUM)

| Scenario | Current Behavior | Expected |
|----------|------------------|----------|
| Server returns HTTP 400 with error detail | Sets `api_val = "HTTP 400"` | Should capture and display the `detail` field |
| Backend raises exception, API returns 400 | Direct path shows `"Err: ..."`, API path shows `"HTTP 400"` | Both should show comparable error messages for diagnosis |
| Server unreachable (connection refused) | Sets `api_val = "Conn Err..."` but continues silently | Should warn loudly and offer to abort |

**Fix:** 
- Capture `res.json().get("detail")` on non-200 responses
- Add a server connectivity check before the test loop
- Distinguish between "expected error" (both sides fail same way) vs "unexpected divergence"

---

### 3.8 Missing Test Categories (LOW)

| Category | Status |
|----------|--------|
| Parametric cases from `generate_parametric_cases()` | ✅ Generated and extended into TEST_CASES |
| Non-default `variable` parameter | ❌ Not tested |
| `use_mpmath=True` high-precision mode | ❌ Not tested |
| Explicit `n` parameter in `/integrate` | ❌ Always relies on auto-analysis |
| Error responses (invalid expressions) | ❌ Not tested |

---

## 4. Proposed Changes to `comprehensive_grok_suite_Gr0.py`

### Phase 1: Structural Fixes (Must Do)

#### 4.1 Fix sys.path import
```python
# BEFORE (line 9):
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# AFTER:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

#### 4.2 Make API URL configurable
```python
import argparse  # or use os.environ.get()
API_URL = os.environ.get("QUADRATURE_API_URL", "http://localhost:8000")
```

#### 4.3 Add server health check before test loop
```python
def check_server():
    """Verify API is reachable and handles infinity intervals."""
    try:
        res = requests.post(f"{API_URL}/integrate", json={
            "expression": "exp(-x**2)",
            "interval": ["-Infinity", "Infinity"]  # test string infinity round-trip
        }, timeout=5)
        if res.status_code == 200 and res.json().get("value") is not None:
            return True, "OK — server handles infinity intervals"
        elif res.status_code == 200:
            return True, "WARNING — server returned value=None for infinite interval"
        else:
            return False, f"HTTP {res.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API server"
```

### Phase 2: Enhanced Comparison (Should Do)

#### 4.4 Expand response field validation
Modify `run_comparison()` to capture and compare ALL response fields:

```python
# New table columns:
table.field_names = [
    "Test", "Family", "n(min)", "Converged", 
    "Value(Dir)", "Value(API)", "ErrEst(API)", "Match?"
]
```

Compare additional fields:
- `suggested_min_n` (analyze response)
- `converged` (integrate response)  
- `error_estimate` (integrate response, with tolerance-aware comparison)
- `n_nodes` (integrate response)

#### 4.5 Improve error message capture on API failures
```python
# BEFORE:
api_val = f"HTTP {res_i.status_code}"

# AFTER:
detail = res_i.json().get("detail", "") if res_i.status_code == 400 else ""
api_val = f"HTTP {res_i.status_code}: {detail[:40]}"
```

### Phase 3: Extended Test Coverage (Nice to Have)

#### 4.6 Add non-default parameter test cases
```python
# Test with explicit n parameter
{"name": "Leg_ExplicitN", "expr": "x**2", "interval": (-1, 1), 
 "target_family": "Legendre", "integrate_n": 16},

# Test with use_mpmath=True  
{"name": "Herm_HighPrec", "expr": "exp(-x**2)", "interval": (-inf, inf),
 "target_family": "Hermite", "use_mpmath": True},
```

#### 4.7 Add error-handling test cases
```python
ERROR_TEST_CASES = [
    {"name": "Err_InvalidExpr", "expr": "xyz_invalid()", "expect_error": True},
]
```

---

## 5. Files Modified vs. Files Protected

| File | Action |
|------|--------|
| `api/comprehensive_grok_suite_Gr0.py` | ✅ MODIFY (target of this plan) |
| `api/quadrature_analyzer.py` | 🔒 NEVER MODIFY |
| `legendre/**` | 🔒 NEVER MODIFY |
| `chebyshev/**` | 🔒 NEVER MODIFY |
| `hermite/**` | 🔒 NEVER MODIFY |
| `laguerre/**` | 🔒 NEVER MODIFY |
| `api/main.py` | 🔒 NOT MODIFIED (server under test) |
| `api/main_mod4intv.py` | 🔒 NOT MODIFIED (server under test) |

---

## 6. Implementation Checklist

- [ ] **Phase 1 — Structural Fixes**
  - [ ] Fix sys.path to use project root (2-line change)
  - [ ] Make API_URL configurable via environment variable
  - [ ] Add `check_server()` pre-flight health check
  - [ ] Improve error capture on HTTP non-200 responses

- [ ] **Phase 2 — Enhanced Comparison**
  - [ ] Expand PrettyTable columns to include more response fields
  - [ ] Compare `suggested_min_n`, `converged`, `n_nodes` between paths
  - [ ] Add tolerance-aware comparison for `error_estimate`
  - [ ] Update fidelity scoring to weight field matches

- [ ] **Phase 3 — Extended Coverage**
  - [ ] Add test cases with explicit `n` parameter
  - [ ] Add test case with `use_mpmath=True`
  - [ ] Add error-handling test cases (invalid expressions)
  - [ ] Document known limitations in module docstring

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| sys.path change breaks import on some systems | Low | Test on both Windows and Linux paths |
| Health check adds latency to test startup | Negligible | Single HTTP call, ~50ms |
| Expanded comparison reveals real middleware bugs | Medium | That is the POINT of the test suite |
| String infinity handling differs between server variants | High (if wrong server used) | Health check detects this immediately |

---

## 8. Verification Criteria

After implementing all Phase 1 + Phase 2 changes:

1. Test suite runs against `main_mod4intv.py` and reports **100% fidelity** on all existing test cases
2. Pre-flight health check passes (server reachable, infinity intervals handled)
3. All new response fields (`converged`, `n_nodes`, `error_estimate`) match between direct and API paths
4. Error messages from API 400 responses are captured and displayed

---

*End of Plan*