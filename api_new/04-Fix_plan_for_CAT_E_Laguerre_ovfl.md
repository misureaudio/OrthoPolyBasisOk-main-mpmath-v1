The fix plan for CATEGORY E (Laguerre catastrophic overflow) has been written to `LAGUERRE_OVERFLOW_FIX_PLAN.md`.

**Summary of the analysis:**

1. **Root cause identified:** `_integrate_laguerre()` (lines 1134-1142) strips the integrand by dividing by e^(-x), but for decay rates c < 1.0, this produces a stripped function that grows exponentially — causing overflow at large Laguerre nodes. Unlike `_integrate_hermite()`, there is no sanity check or fallback mechanism.

2. **The fix (3 changes in a new copy file):**
   - Add `_compute_effective_support_laguerre()` — computes the finite window [0, L] capturing ≥(1-ε) of integral mass for semi-infinite integrals
   - Replace `_integrate_laguerre()` with overflow-safe version: pre-computation probe at x∈{10, 30, 60} → growth detection → fallback to finite-window Legendre on [0, L]
   - Post-computation non-finite result check as a safety net

3. **Design principle:** Mirror the existing Hermite fallback pattern (lines 1107-1126) — strip → sanity check → fall back to finite-window Legendre.

4. **No existing files modified.** The fix is applied only to a new copy: `api_new/quadrature_analyzer_laguerre_fixed.py`.

The plan document includes detailed code snippets, mathematical verification (analytical result 10/3 for the failing test case), and a testing strategy covering all Laguerre-related test cases.