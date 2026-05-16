Created `api_new/quadrature_analyzer_d_adapted_v2.py` with CATEGORY C Gauss-Jacobi Weight Matching fix implemented according to the plans in documents 10 and 11.

**Changes made (no files in chebyshev/, hermite/, laguerre/, legendre/ were modified):**

1. **FunctionAnalysis fields added:**
   - `left_singularity_alpha: float` — exponent of (x-a)^(-alpha) at left endpoint
   - `right_singularity_beta: float` — exponent of (b-x)^(-beta) at right endpoint  
   - `has_algebraic_endpoint_singularity: bool`

2. **New method `_extract_endpoint_singularity_exponents()`:** Numerically detects algebraic endpoint singularities by fitting log|f(x)| vs log|x-endpoint| using geometrically spaced probe points and linear regression. Returns {'left': alpha, 'right': beta}.

3. **New method `_integrate_jacobi()`:** Gauss-Jacobi quadrature using `scipy.special.roots_jacobi`. Maps [a,b] to [-1,1], applies proper weight scaling factor scale^(1-alpha-beta).

4. **New method `_integrate_with_jacobi_weight()`:** Strips singular weight from f(x) by multiplying with (x-a)^alpha*(b-x)^beta, then calls _integrate_jacobi.

5. **Modified `execute_quadrature()`:** When algebraic endpoint singularities detected on finite intervals, routes through Jacobi path instead of standard family.

**Test results:**
- `1/sqrt(1-x^2)` on [-1,1]: alpha=0.4977, beta=0.4977 detected (true: 0.5), result → π ≈ 3.14155 at n=256
- `(1-x²)^(-0.3)*exp(0.3x)` on [-1,1]: alpha=0.2959, beta=0.3013 detected (true: 0.3), converging toward ~2.553
- Non-singular cases (`exp(-x²)`, `exp(-x²)*sin(x)`) unaffected — still converge correctly