Category C plan written to `api_new/CATEGORY_C_ENDPOINT_SINGULARITIES_PLAN.md`. No code was modified.

**Key Findings:**

1. **Leg_Sing_Alg** `(1-x²)^(-0.3)·exp(0.3x)` and **Cheb_Sing_Beta** `(1-x²)^(-0.4)·cos(5x)` have genuine algebraic endpoint singularities where f(x) → ∞ at x=±1. Gauss-Legendre converges only as O(n^(-p)) for these — even n=512 gives error ~10⁻⁴, far from tol=1e-12.

2. **Stress_NarrowPeak** is a false alarm — it's not a true singularity and already converges at n=64 to 15 digits. The N/N flag comes from the convergence check using too-low initial n (32→64).

3. These currently pass fidelity tests (both Direct and API return matching N/N), so Category C is about improving absolute accuracy, not fixing proxy mismatches.

**Recommended Fix: Gauss-Jacobi Weight Matching**
- Detect algebraic singularity exponents α from expression structure
- Strip the singular weight (1-x²)^(-α) and use Gauss-Jacobi quadrature with parameters (α,α) for the remaining smooth function g(x)
- This restores exponential convergence instead of O(n^(-p))

**Alternative Options Documented:** Variable transformation (x=cos(t)), interval splitting with adaptive refinement, or honest non-convergence reporting.