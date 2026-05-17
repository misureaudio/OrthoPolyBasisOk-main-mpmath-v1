# Test Suite Extension Plan for `comprehensive_grok_suite_Gr0.py`

**Objective:** Extend the existing Backend-vs-Middleware fidelity test suite to exercise all new mpmath endpoints and enriched response fields introduced by `main_mod4intv_mpmath.py`.

**Constraint:** The original `comprehensive_grok_suite_Gr0.py` is NOT modified. A new file `comprehensive_grok_suite_Gr1.py` will be created as the extended version. No polynomial modules (chebyshev, hermite, laguerre, legendre) are modified.

---

## 1. Current State Analysis

### What the Existing Suite Already Does Well

| Capability | Status | Location |
|------------|--------|----------|
| Backend vs API fidelity comparison | ✅ Complete | `run_comparison()` lines 263-407 |
| Interval sanitization (inf/-inf → strings) | ✅ Complete | `_format_api_interval()` line 149 |
| NaN/None-aware float comparison | ✅ Complete | `float_eq()`, `values_match()` lines 164-215 |
| Server health check with infinity round-trip | ✅ Complete | `check_server()` lines 222-257 |
| Extended test cases (explicit n, use_mpmath) | ✅ Partial | `EXTENDED_TEST_CASES` line 126 |
| Error handling tests | ✅ Basic | `ERROR_TEST_CASES` line 140 |

### What Is Missing for the New API

| Gap # | Description | Impact |
|-------|-------------|--------|
| G1 | No test of `/integrate/compare` endpoint | Cannot verify side-by-side comparison functionality |
| G2 | No test of `/integrate/high-precision` endpoint | Cannot verify convenience HP endpoint |
| G3 | Enriched `/analyze` response fields not validated | Category C (singularity exponents) and decay fields untested |
| G4 | `tol` parameter passthrough not tested | Cannot verify custom tolerance works through API |
| G5 | No mpmath-specific accuracy tests | Cannot verify that mpmath actually produces more accurate results than float64 |
| G6 | Server health check only targets old server (`main_mod4intv.py`) | New server name `main_mod4intv_mpmath.py` not referenced |
| G7 | No test of diagnostic response fields (`mpmath_requested`, `tolerance_used`) | Cannot verify new response metadata is correct |

---

## 2. Modification Plan

### Phase 1: Infrastructure Updates (New File)

#### P1.1 Create New Test Suite File

Create `comprehensive_grok_suite_Gr1.py` as a copy of the existing suite with all extensions applied. The original file remains untouched for backward compatibility.

**Header update:**
```python
"""Extended Test Suite: Backend vs MPMath-Enhanced Middleware Fidelity Check.

Validates that main_mod4intv_mpmath.py faithfully proxies requests to the
quadrature_analyzer backend AND correctly exposes all new mpmath endpoints.

Requirements:
    - Server must be `main_mod4intv_mpmath.py` (has mpmath extensions).
      Start with:  python -m api_new.main_mod4intv_mpmath
"""
```

#### P1.2 Update Server Health Check

Extend `check_server()` to verify the new endpoints exist:

**Before:**
```python
def check_server():
    """Verify API is reachable and handles infinity intervals correctly."""
    try:
        res = requests.post(f"{API_URL}/integrate", json={...}, timeout=5)
        # ... only checks /integrate
```

**After (additions):**
```python
def check_server():
    """Verify API is reachable, handles infinity intervals, and has new endpoints."""
    results = []

    # Original: basic integration test
    res = requests.post(f"{API_URL}/integrate", json={...}, timeout=5)
    if res.status_code == 200:
        results.append(("basic_integrate", True))
    else:
        results.append(("basic_integrate", False))

    # NEW: verify /integrate/compare exists
    try:
        res = requests.post(f"{API_URL}/integrate/compare", json={...}, timeout=5)
        results.append(("compare_endpoint", res.status_code == 200))
    except Exception:
        results.append(("compare_endpoint", False))

    # NEW: verify /integrate/high-precision exists
    try:
        res = requests.post(f"{API_URL}/integrate/high-precision", json={...}, timeout=5)
        results.append(("high_precision_endpoint", res.status_code == 200))
    except Exception:
        results.append(("high_precision_endpoint", False))

    # NEW: verify enriched /analyze response has new fields
    try:
        res = requests.post(f"{API_URL}/analyze", json={...}, timeout=5)
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

    return all(ok for _, ok in results), "; ".join(f"{name}={'OK' if ok else 'FAIL'}" for name, ok in results)
```

### Phase 2: New Test Case Categories

#### P2.1 Tolerance Passthrough Tests

Verify that the `tol` parameter is correctly passed through to the backend:

```python
TOLERANCE_TEST_CASES = [
    # Standard tolerance (default 1e-12)
    {"name": "Tol_Default",        "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": None,                  "expected_tol_used": 1e-12},

    # Tight tolerance with mpmath
    {"name": "Tol_Tight_mpmath",   "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 1e-20,                 "use_mpmath": True,   "expected_tol_used": 1e-20},

    # Loose tolerance (should converge faster)
    {"name": "Tol_Loose",          "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 1e-6,                  "expected_tol_used": 1e-6},

    # Invalid tolerance (should return HTTP 400)
    {"name": "Tol_Invalid_Zero",   "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 0.0,                   "expect_error": True},

    # Invalid tolerance (too large)
    {"name": "Tol_Invalid_Large",  "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "tol": 5.0,                   "expect_error": True},
]
```

#### P2.2 Compare Endpoint Tests

Verify the `/integrate/compare` endpoint returns correct structure and meaningful comparisons:

```python
COMPARE_TEST_CASES = [
    # Smooth function — should show high agreement (use_numpy recommendation)
    {"name": "Compare_Smooth",     "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "expected_recommendation": "use_numpy"},

    # Endpoint singularity — may show lower agreement (use_mpmath or investigate)
    {"name": "Compare_Singular",   "expr": "1/sqrt(x)", "interval": (0.0, 1.0),
     "expected_min_digits": 6},

    # Infinite domain Hermite — should converge well in both modes
    {"name": "Compare_Hermite",    "expr": "exp(-x**2)", "interval": None,
     "expected_recommendation": "use_numpy"},

    # Semi-infinite Laguerre
    {"name": "Compare_Laguerre",   "expr": "exp(-0.5*x)", "interval": (0.0, math.inf),
     "expected_min_digits": 10},
]
```

#### P2.3 High-Precision Endpoint Tests

Verify the `/integrate/high-precision` endpoint:

```python
HIGH_PRECISION_TEST_CASES = [
    # Basic HP integration
    {"name": "HP_Basic",           "expr": "exp(-x**2)", "interval": (-1.0, 1.0),
     "expected_mpmath_used": True},

    # HP with custom tolerance
    {"name": "HP_Custom_Tol",      "expr": "sin(x)/x", "interval": (0.0, 1.0),
     "tol": 1e-25,                 "expected_mpmath_used": True},

    # HP on infinite domain
    {"name": "HP_Infinite",        "expr": "exp(-x**2)", "interval": None,
     "expected_mpmath_used": True},
]
```

#### P2.4 Enriched Analysis Response Tests

Verify new fields in `/analyze` response:

```python
ANALYSIS_ENRICHMENT_CASES = [
    # Finite interval — should have interval_type="finite"
    {"name": "Analyze_Finite",     "expr": "exp(x)", "interval": (-1.0, 1.0),
     "expected_interval_type": "finite"},

    # Infinite domain — should detect gaussian decay
    {"name": "Analyze_Gaussian",   "expr": "exp(-x**2)", "interval": None,
     "expected_decay_type": "gaussian",
     "expected_hermite_compatible": True},

    # Endpoint singularity — Category C detection
    {"name": "Analyze_Singularity","expr": "1/sqrt(x)", "interval": (0.0, 1.0),
     "expected_has_algebraic_endpoint_singularity": True},

    # Semi-infinite with exponential decay
    {"name": "Analyze_Exponential","expr": "exp(-x)", "interval": None,
     "expected_decay_type": "exponential"},
]
```

#### P2.5 Mpmath Accuracy Verification Tests

Verify that mpmath actually produces more accurate results than float64 for challenging cases:

```python
MPMATH_ACCURACY_CASES = [
    # Near-singular integrand where float64 may lose precision
    {"name": "Acc_NearSingular",   "expr": "(1-x**2)**(-0.49)", "interval": (-1.0, 1.0),
     "expected_mpmath_better": True},

    # High-frequency oscillation where cancellation matters
    {"name": "Acc_HighFreq",       "expr": "sin(80*x)*exp(-x**2/4)", "interval": (-5.0, 5.0),
     "expected_mpmath_better": True},

    # Smooth function — both should agree (no mpmath advantage)
    {"name": "Acc_Smooth",         "expr": "exp(0.5*x)", "interval": (-1.0, 1.0),
     "expected_agreement_digits_min": 14},
]
```

### Phase 3: New Test Runner Functions

#### P3.1 Tolerance Passthrough Test Runner

```python
def run_tolerance_tests():
    """Test that tol parameter is correctly passed through to backend."""
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
                status = "✅" if ok else f"❌ (got HTTP {res.status_code})"
            else:
                data = res.json()
                tol_used = data.get("tolerance_used")
                expected_tol = tc["expected_tol_used"]
                ok = abs(tol_used - expected_tol) < 1e-30 if tol_used is not None else False
                status = "✅" if ok else f"❌ (got tol={tol_used}, expected {expected_tol})"

            if ok:
                passed += 1
            print(f"  {status} {name}")

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\nTolerance Tests: {passed}/{len(TOLERANCE_TEST_CASES)} passed.")
```

#### P3.2 Compare Endpoint Test Runner

```python
def run_compare_tests():
    """Test /integrate/compare endpoint structure and behavior."""
    print(f"\n--- Compare Endpoint Tests ({len(COMPARE_TEST_CASES)} cases) ---\n")
    passed = 0

    for tc in COMPARE_TEST_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc.get("interval"))}

        try:
            res = requests.post(f"{API_URL}/integrate/compare", json=payload, timeout=60)

            if res.status_code != 200:
                print(f"  ❌ {name}: HTTP {res.status_code}")
                continue

            data = res.json()

            # Verify required fields exist
            required_fields = ["numpy_result", "mpmath_result", "value_difference",
                              "agreement_digits", "recommendation"]
            has_all_fields = all(f in data for f in required_fields)

            if not has_all_fields:
                missing = [f for f in required_fields if f not in data]
                print(f"  ❌ {name}: Missing fields: {missing}")
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

            status = "✅" if ok else f"❌ {reason}"
            if ok:
                passed += 1
            print(f"  {status} {name}: digits={data['agreement_digits']}, rec={data['recommendation']}")

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\nCompare Tests: {passed}/{len(COMPARE_TEST_CASES)} passed.")
```

#### P3.3 High-Precision Endpoint Test Runner

```python
def run_high_precision_tests():
    """Test /integrate/high-precision endpoint."""
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
                print(f"  ❌ {name}: HTTP {res.status_code}")
                continue

            data = res.json()

            # Verify mpmath_used is True
            ok = data.get("mpmath_used") == True
            reason = ""
            if not ok:
                reason = f"(mpmath_used={data.get('mpmath_used')})"

            status = "✅" if ok else f"❌ {reason}"
            if ok:
                passed += 1
            print(f"  {status} {name}: value={data.get('value', 'N/A'):.6e}")

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\nHigh-Precision Tests: {passed}/{len(HIGH_PRECISION_TEST_CASES)} passed.")
```

#### P3.4 Enriched Analysis Test Runner

```python
def run_analysis_enrichment_tests():
    """Test enriched /analyze response fields."""
    print(f"\n--- Analysis Enrichment Tests ({len(ANALYSIS_ENRICHMENT_CASES)} cases) ---\n")
    passed = 0

    for tc in ANALYSIS_ENRICHMENT_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc.get("interval"))}

        try:
            res = requests.post(f"{API_URL}/analyze", json=payload, timeout=10)

            if res.status_code != 200:
                print(f"  ❌ {name}: HTTP {res.status_code}")
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

            status = "✅" if ok else f"❌ ({'; '.join(reasons)})"
            if ok:
                passed += 1
            print(f"  {status} {name}")

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\nAnalysis Enrichment Tests: {passed}/{len(ANALYSIS_ENRICHMENT_CASES)} passed.")
```

#### P3.5 Mpmath Accuracy Verification Test Runner

```python
def run_mpmath_accuracy_tests():
    """Verify mpmath produces more accurate results than float64 for challenging cases."""
    print(f"\n--- MPMath Accuracy Tests ({len(MPMATH_ACCURACY_CASES)} cases) ---\n")
    passed = 0

    for tc in MPMATH_ACCURACY_CASES:
        name = tc["name"]
        payload = {"expression": tc["expr"], "interval": _format_api_interval(tc.get("interval"))}

        try:
            # Use compare endpoint to get both results
            res = requests.post(f"{API_URL}/integrate/compare", json=payload, timeout=60)

            if res.status_code != 200:
                print(f"  ❌ {name}: HTTP {res.status_code}")
                continue

            data = res.json()
            digits = data.get("agreement_digits", 0)
            rec = data.get("recommendation", "unknown")

            ok = True
            reason = ""

            if tc.get("expected_mpmath_better"):
                # mpmath should be better → agreement < 14 digits or recommendation is use_mpmath/investigate
                ok = (digits < 14) or rec in ("use_mpmath", "investigate")
                reason = f"(digits={digits}, rec={rec})"

            if tc.get("expected_agreement_digits_min"):
                # Both should agree well → digits >= threshold
                ok = digits >= tc["expected_agreement_digits_min"]
                reason = f"(digits={digits}, expected >= {tc['expected_agreement_digits_min']})"

            status = "✅" if ok else f"❌ {reason}"
            if ok:
                passed += 1
            print(f"  {status} {name}: digits={digits}, rec={rec}")

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\nMPMath Accuracy Tests: {passed}/{len(MPMATH_ACCURACY_CASES)} passed.")
```

### Phase 4: Main Function Update

Update `run_comparison()` to call all new test runners and produce a consolidated score:

**Before:**
```python
if __name__ == "__main__":
    run_comparison()
```

**After:**
```python
def run_all_tests():
    """Run ALL test suites and produce consolidated report."""
    scores = {}

    # Original fidelity tests (unchanged)
    print("=" * 70)
    print("PHASE 1: Backend vs Middleware Fidelity")
    print("=" * 70)
    scores["fidelity"] = run_comparison()   # returns passed/total

    # New mpmath-specific tests
    print("\n" + "=" * 70)
    print("PHASE 2: Tolerance Passthrough")
    print("=" * 70)
    scores["tolerance"] = run_tolerance_tests()

    print("\n" + "=" * 70)
    print("PHASE 3: Compare Endpoint")
    print("=" * 70)
    scores["compare"] = run_compare_tests()

    print("\n" + "=" * 70)
    print("PHASE 4: High-Precision Endpoint")
    print("=" * 70)
    scores["high_precision"] = run_high_precision_tests()

    print("\n" + "=" * 70)
    print("PHASE 5: Analysis Enrichment")
    print("=" * 70)
    scores["enrichment"] = run_analysis_enrichment_tests()

    print("\n" + "=" * 70)
    print("PHASE 6: MPMath Accuracy Verification")
    print("=" * 70)
    scores["accuracy"] = run_mpmath_accuracy_tests()

    # Consolidated report
    print("\n" + "=" * 70)
    print("CONSOLIDATED REPORT")
    print("=" * 70)
    total_passed = sum(s.split("/")[0] for s in scores.values())
    total_tests = sum(s.split("/")[1] for s in scores.values())
    # ... formatted table of all phase results

if __name__ == "__main__":
    run_all_tests()
```

---

## 3. Summary of Changes

| Phase | Change | Description | New Lines |
|-------|--------|-------------|-----------|
| P1.1 | Create new file | `comprehensive_grok_suite_Gr1.py` as extended version | ~50 (header + imports) |
| P1.2 | Update health check | Verify all 4 endpoints exist and enriched fields present | ~60 |
| P2.1 | Tolerance test cases | 5 new test cases for tol passthrough | ~30 |
| P2.2 | Compare endpoint tests | 4 new test cases for /integrate/compare | ~25 |
| P2.3 | HP endpoint tests | 3 new test cases for /integrate/high-precision | ~20 |
| P2.4 | Analysis enrichment tests | 4 new test cases for enriched response fields | ~25 |
| P2.5 | Mpmath accuracy tests | 3 new test cases comparing NumPy vs mpmath precision | ~20 |
| P3.1 | Tolerance runner | New test runner function | ~40 |
| P3.2 | Compare runner | New test runner function | ~60 |
| P3.3 | HP runner | New test runner function | ~40 |
| P3.4 | Enrichment runner | New test runner function | ~50 |
| P3.5 | Accuracy runner | New test runner function | ~50 |
| P4 | Main function update | Consolidated multi-phase test execution | ~40 |

**Total estimated new lines:** ~510  
**Original file preserved:** Yes (no modifications to `comprehensive_grok_suite_Gr0.py`)  
**New total test cases:** 37 original + 19 new = **56 tests across 6 phases**

---

## 4. Execution Instructions

```bash
# Start the mpmath-extended server
python -m api_new.main_mod4intv_mpmath

# Run the extended test suite (in another terminal)
QUADRATURE_API_URL=http://localhost:8000 python api_new/comprehensive_grok_suite_Gr1.py
```

---

## 5. Expected Output Structure

```
======================================================================
PHASE 1: Backend vs Middleware Fidelity
======================================================================
API Server [http://localhost:8000]: OK — all endpoints available; enriched fields present

Running Comprehensive Test Suite: Backend vs Middleware (37 tests)
[... original fidelity table ...]
Fidelity Score: 35/37 tests passed 1-to-1 mapping.

======================================================================
PHASE 2: Tolerance Passthrough
======================================================================
--- Tolerance Tests (5 cases) ---
  ✅ Tol_Default
  ✅ Tol_Tight_mpmath
  ✅ Tol_Loose
  ✅ Tol_Invalid_Zero
  ✅ Tol_Invalid_Large
Tolerance Tests: 5/5 passed.

======================================================================
PHASE 3: Compare Endpoint
======================================================================
--- Compare Endpoint Tests (4 cases) ---
  ✅ Compare_Smooth: digits=16, rec=use_numpy
  ✅ Compare_Singular: digits=8, rec=use_mpmath
  ...
Compare Tests: 4/4 passed.

... [remaining phases] ...

======================================================================
CONSOLIDATED REPORT
======================================================================
Phase              Score    Status
─────────────────────────────────────
Fidelity           35/37    ✅ PASS
Tolerance          5/5      ✅ PASS
Compare            4/4      ✅ PASS
High-Precision     3/3      ✅ PASS
Enrichment         4/4      ✅ PASS
Accuracy           3/3      ✅ PASS
─────────────────────────────────────
TOTAL              54/56    ✅ PASS (96.4%)