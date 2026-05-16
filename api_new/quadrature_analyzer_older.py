# quadrature_analyzer.py
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
from sympy import (
    Symbol, diff, N, S, pi, log, sin, cos, tan, exp, sqrt,
    Abs as SymAbs, limit, solve, Derivative, Mul, Pow,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

_PARSER_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


class PolynomialFamily(str, Enum):
    LEGENDRE = "Legendre"
    CHEBYSHEV = "Chebyshev"
    HERMITE = "Hermite"
    LAGUERRE = "Laguerre"


@dataclass
class QuadratureResult:
    value: float
    family_used: PolynomialFamily
    n_nodes: int
    converged: bool
    error_estimate: Optional[float] = None
    message: str = ""


@dataclass
class FunctionAnalysis:
    original_expr: str
    sympy_expr: object
    variable: str = "x"
    interval_type: str = "finite"  # finite | semi_infinite | infinite
    singularities: list = field(default_factory=list)
    has_endpoint_singularity: bool = False
    has_interior_singularity: bool = False
    max_deriv_order_probed: int = 6
    derivative_growth_rate: str = "bounded"  # bounded | polynomial | exponential | super_exponential
    decay_type: str = "none"  # none | gaussian | exponential | algebraic | oscillatory
    decay_rate: float = 0.0
    is_periodic_on_interval: bool = False
    approximate_period: Optional[float] = None
    recommended_family: PolynomialFamily = PolynomialFamily.LEGENDRE
    confidence: str = "high"
    recommendation_reason: str = ""
    suggested_min_n: int = 8
    suggested_max_n: int = 64
    degree_criteria: list[str] = field(default_factory=list)
    # Actual interval used (for recommend_usage and execute_quadrature)
    interval_a: float = -1.0
    interval_b: float = 1.0


def _safe_eval_sympy(expr, x_val):
    try:
        return float(N(expr.subs("x", x_val), 15))
    except Exception:
        return float("nan")


class QuadratureAnalyzer:
    """
    Reads a mathematical expression (string or SymPy object) that defines a real
    function of one variable, analyses its properties, and recommends the optimal
    orthogonal-polynomial family together with quadrature-degree criteria.

    The recommended polynomial family maps directly to the corresponding module in
    OrthoPolyB_np_mp:
      - Legendre  -> `from legendre import LegendreQuadrature`
      - Chebyshev -> `from chebyshev import ChebyshevQuadrature`
      - Hermite   -> `from hermite import GaussHermiteQuadrature`
      - Laguerre  -> `from laguerre import LaguerreQuadrature`

    Usage:
        analyzer = QuadratureAnalyzer()
        analysis = analyzer.analyze("exp(-x**2) * sin(x)", interval=(-1, 1))
        print(analysis.recommended_family)
        # Or compute directly:
        result = analyzer.execute_quadrature("exp(-x**2)")
        print(result.value)
    """

    def __init__(self, default_variable: str = "x"):
        self._var = Symbol(default_variable)

    def analyze(
        self,
        expression,
        interval=None,
        *,
        variable: str = "x",
        max_deriv_order: int = 6,
    ) -> FunctionAnalysis:
        self._var = Symbol(variable)
        expr = self._parse(expression, variable)
        if interval is None:
            interval = self._infer_interval(expr, variable)
        a, b = float(interval[0]), float(interval[1])

        analysis = FunctionAnalysis(
            original_expr=str(expression),
            sympy_expr=expr,
            variable=variable,
            interval_a=a,
            interval_b=b,
        )
        analysis.interval_type = self._classify_interval(a, b)
        analysis.singularities = self._find_singularities(expr, a, b)
        analysis.has_endpoint_singularity = any(
            s["location"] in (a, b) for s in analysis.singularities
        )
        analysis.has_interior_singularity = any(
            a < s["location"] < b for s in analysis.singularities
        )
        analysis.derivative_growth_rate = self._probe_derivative_growth(expr, a, b, max_deriv_order)
        if analysis.interval_type in ("semi_infinite", "infinite"):
            analysis.decay_type, analysis.decay_rate = self._probe_decay(expr, variable)
        # Periodicity: only probe on finite intervals
        if not np.isinf(a) and not np.isinf(b):
            analysis.is_periodic_on_interval, analysis.approximate_period = (
                self._probe_periodicity(expr, a, b)
            )
        rec = self._recommend_family(analysis, a, b)
        analysis.recommended_family = rec.family
        analysis.confidence = rec.confidence
        analysis.recommendation_reason = rec.reason
        analysis.degree_criteria = self._degree_criteria(analysis)
        lo, hi = self._suggest_degree_range(analysis)
        analysis.suggested_min_n = lo
        analysis.suggested_max_n = hi
        return analysis

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse(expression, variable: str) -> object:
        if isinstance(expression, str):
            return parse_expr(
                expression,
                local_dict={variable: Symbol(variable)},
                transformations=_PARSER_TRANSFORMS,
            )
        return expression

    @staticmethod
    def _infer_interval(expr, variable: str) -> tuple:
        """
        Heuristic interval inference from the expression structure.

        Recognised patterns (checked in priority order):
          - exp(-c*x^2)  => (-inf, +inf)
          - x^n * exp(-x) or similar with positive x powers and exp(-x) => [0, inf)
          - log(x)       => [0, inf)
          - Default      => (-1, 1)
        """
        v = Symbol(variable)

        # Pattern 1: Gaussian decay exp(-c*x^2), c > 0
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_quadratic(inner, v):
                    return (-float("inf"), float("inf"))
            except Exception:
                pass

        # Pattern 2: exp(-c*x) with c > 0 and no other infinite-domain indicators
        # This catches exp(-x), x^2*exp(-x), etc.
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_linear(inner, v):
                    return (0.0, float("inf"))
            except Exception:
                pass

        # Pattern 3: log(x) => [0, inf)
        for sub in expr.atoms(log):
            try:
                arg = sub.args[0]
                if arg == v or (hasattr(arg, "is_Mul") and arg.is_Mul and len(arg.args) == 1 and arg.args[0] == v):
                    return (0.0, float("inf"))
            except Exception:
                pass

        # Pattern 4: 1/sqrt(x) or x^(-p) with p > 0 => [0, inf)
        try:
            den = expr.as_numer_denom()[1]
            if den.has(v):
                for pt in den.atoms():
                    if hasattr(pt, "is_Pow") and pt.is_Pow and pt.exp < 0:
                        # Check if the base is just v (or a positive multiple)
                        if pt.base == v or (hasattr(pt.base, "is_Mul") and pt.base.is_Mul):
                            return (0.0, float("inf"))
        except Exception:
            pass

        # Default: finite interval
        return (-1.0, 1.0)

    @staticmethod
    def _is_negative_quadratic(expr, v) -> bool:
        """Check if expr is a negative quadratic form in v, e.g. -c*v^2 or -v^2/2."""
        try:
            # Expand to handle cases like -(x**2)/2
            expanded = expr.expand()
            # Collect all terms that contain v
            v_terms = [t for t in expanded.atoms() if t.has(v)]
            # Also check the top-level expression structure
            if hasattr(expr, "is_Mul") and expr.is_Mul:
                args_list = list(expr.args)
                coeff_parts = []
                has_x2 = False
                for a in args_list:
                    if hasattr(a, "is_Pow") and a.is_Pow:
                        try:
                            if a.base == v and a.exp == 2:
                                has_x2 = True
                        except Exception:
                            pass
                    elif not a.has(v):
                        coeff_parts.append(a)
                if has_x2 and coeff_parts:
                    coeff = float(Mul(*coeff_parts))
                    return coeff < 0
            # Check for -(v**2) or similar direct patterns
            neg_v2 = -v ** 2
            if expr == neg_v2:
                return True
            # Check expanded form: should be a single term c*v^2 with c < 0
            if hasattr(expanded, "is_Mul") and expanded.is_Mul:
                args_list = list(expanded.args)
                coeff_parts = []
                has_x2 = False
                for a in args_list:
                    if hasattr(a, "is_Pow") and a.is_Pow:
                        try:
                            if a.base == v and a.exp == 2:
                                has_x2 = True
                        except Exception:
                            pass
                    elif not a.has(v):
                        coeff_parts.append(a)
                if has_x2 and coeff_parts:
                    coeff = float(Mul(*coeff_parts))
                    return coeff < 0
            # Check for Pow: -v**2 is actually (-1)*v**2, but sometimes represented as -(v**2)
            if hasattr(expr, "is_Pow") and expr.is_Pow:
                try:
                    if expr.base == v and expr.exp == 2:
                        return False  # positive x^2
                except Exception:
                    pass
        except Exception:
            pass
        return False

    @staticmethod
    def _is_negative_linear(expr, v) -> bool:
        """Check if expr is -c*x with c > 0, e.g. -x or -2*x."""
        try:
            if expr == -v:
                return True
            if hasattr(expr, "is_Mul") and expr.is_Mul:
                args_list = list(expr.args)
                coeff_parts = []
                has_v = False
                for a in args_list:
                    if a == v:
                        has_v = True
                    elif not a.has(v):
                        coeff_parts.append(a)
                if has_v and coeff_parts:
                    coeff = float(Mul(*coeff_parts))
                    return coeff < 0
        except Exception:
            pass
        return False

    @staticmethod
    def _classify_interval(a: float, b: float) -> str:
        if a == -float("inf") and b == float("inf"):
            return "infinite"
        if a == -float("inf") or b == float("inf"):
            return "semi_infinite"
        return "finite"

    def _find_singularities(self, expr, a: float, b: float) -> list:
        v = self._var
        found: list = []
        try:
            den = expr.as_numer_denom()[1]
            if den != 1:
                roots = solve(den, v, domain=S.Complexes)
                for r in roots:
                    try:
                        rv = float(N(r, 15))
                        if math.isnan(rv):
                            continue
                        if a <= rv <= b:
                            found.append({"location": rv, "kind": "pole"})
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
        for sub in expr.atoms(log):
            try:
                arg = sub.args[0]
                zero_pts = solve(arg, v)
                for zp in zero_pts:
                    try:
                        zv = float(N(zp, 15))
                        if a <= zv <= b and not math.isnan(zv):
                            found.append({"location": zv, "kind": "log_zero"})
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        for sub in expr.atoms(sqrt):
            try:
                arg = sub.args[0]
                zero_pts = solve(arg, v)
                for zp in zero_pts:
                    try:
                        zv = float(N(zp, 15))
                        if a <= zv <= b and not math.isnan(zv):
                            found.append({"location": zv, "kind": "sqrt_zero"})
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        for sub in expr.atoms(tan):
            try:
                arg = sub.args[0]
                half_pi = float(pi / 2)
                for k in range(-10, 11):
                    sing_val = half_pi + k * float(pi)
                    if a <= sing_val <= b:
                        found.append({"location": sing_val, "kind": "tan_pole"})
            except Exception:
                pass
        unique: list = []
        for s in found:
            if not any(abs(s["location"] - u["location"]) < 1e-8 for u in unique):
                unique.append(s)
        return unique

    def _probe_derivative_growth(self, expr, a: float, b: float, max_order: int) -> str:
        """
        Probe derivative growth at multiple points across the interval.

        Uses Legendre quadrature nodes as probe points (well-distributed,
        including near endpoints). For each point, computes derivatives up to
        order max_order and tracks magnitude growth. Handles zeros gracefully
        by skipping zero values in ratio computation.

        Returns one of: bounded | polynomial | exponential | super_exponential
        """
        v = self._var

        # Determine probe points
        if np.isinf(a) or np.isinf(b):
            # For infinite intervals, use a finite window and transform
            probe_points = [0.0]
            for k in range(1, 6):
                probe_points.append(float(k))
                probe_points.append(-float(k))
        else:
            # Use Gauss-Legendre nodes as well-distributed probe points
            n_probe = min(11, max_order + 2)
            try:
                # from sympy import leggauss
                from numpy import polynomial
                # raw_nodes, _ = leggauss(n_probe)
                raw_nodes, _ = np.polynomial.legendre.leggauss(n_probe)
                # Map [-1, 1] to [a, b]
                scale = (b - a) / 2.0
                shift = (b + a) / 2.0
                probe_points = [float(scale * n_val + shift) for n_val in raw_nodes]
            except ImportError:
                # Fallback: evenly spaced points
                probe_points = np.linspace(a, b, min(11, max_order + 2)).tolist()

        # For each probe point, compute derivative magnitudes and track growth
        all_ratios: list[float] = []
        max_deriv_magnitude: float = 0.0
        zero_count = 0

        for pt in probe_points:
            deriv_magnitudes: list[float] = []
            for k in range(1, min(max_order + 1, 7)):
                dexpr = diff(expr, v, k)
                try:
                    val = float(N(dexpr.subs(v, pt), 10))
                    mag = abs(val)
                    if math.isfinite(mag):
                        deriv_magnitudes.append(mag)
                        max_deriv_magnitude = max(max_deriv_magnitude, mag)
                    else:
                        zero_count += 1
                except Exception:
                    break

            # Compute log-ratios for this probe point, skipping zeros
            if len(deriv_magnitudes) >= 2:
                for i in range(1, len(deriv_magnitudes)):
                    prev = deriv_magnitudes[i - 1]
                    curr = deriv_magnitudes[i]
                    # Skip if prev or curr are near-zero, or if log(prev) would be ~0
                    if prev < 1e-30 or curr < 1e-30:
                        continue
                    log_prev = math.log(prev)
                    if abs(log_prev) < 1e-15: continue
                    ratio = math.log(curr) / log_prev
                    all_ratios.append(ratio)

        if not all_ratios:
            # Not enough data; default to bounded
            return "bounded"

        avg_ratio = np.mean(all_ratios)
        # Use median for robustness against outliers from near-zero derivatives
        median_ratio = float(np.median(all_ratios))

        # Classify based on derivative growth rate
        if avg_ratio < 1.5:
            return "bounded"
        elif avg_ratio < 2.5:
            return "polynomial"   # derivatives grow like k!
        elif avg_ratio < 6.0:
            return "exponential"  # derivatives grow like C^k * k!
        else:
            return "super_exponential"

    # ...

    def _probe_decay(self, expr, variable: str) -> tuple:
        """
        For infinite/semi-infinite intervals, estimate the decay type and rate.

        Improved Gaussian detection handles exp(-c*x^2), exp(-x^2/2), etc.
        by examining the exponent expression structure directly.
        """
        v = Symbol(variable)

        # Check for Gaussian: exp(negative_quadratic_in_x)
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_quadratic(inner, v):
                    # Extract the coefficient of x^2 to get decay rate
                    expanded = inner.expand()
                    coeff = self._extract_x2_coefficient(expanded, v)
                    return ("gaussian", abs(coeff))
            except Exception:
                pass

        # Check for exponential: exp(-c*x) with c > 0
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_linear(inner, v):
                    coeff = self._extract_x_coefficient(inner, v)
                    return ("exponential", abs(coeff))
            except Exception:
                pass

        # Check algebraic decay: x^(-p) in denominator
        try:
            den = expr.as_numer_denom()[1]
            if den.has(v):
                for pt in den.atoms():
                    if hasattr(pt, "is_Pow") and pt.is_Pow and pt.exp < 0:
                        return ("algebraic", float(-pt.exp))
        except Exception:
            pass

        # Oscillatory by default if sin/cos present
        if expr.has(sin) or expr.has(cos):
            return ("oscillatory", 0.0)

        return ("none", 0.0)

    @staticmethod
    def _extract_x2_coefficient(expr, v) -> float:
        """Extract the coefficient c from a term like -c*x^2."""
        try:
            if hasattr(expr, "is_Mul") and expr.is_Mul:
                args_list = list(expr.args)
                coeff_parts = []
                for a in args_list:
                    if hasattr(a, "is_Pow") and a.is_Pow:
                        try:
                            if a.base == v and a.exp == 2:
                                continue
                        except Exception:
                            pass
                    elif not a.has(v):
                        coeff_parts.append(a)
                if coeff_parts:
                    return float(Mul(*coeff_parts))
            # Single term like -x**2 or -3*x**2/2
            if hasattr(expr, "is_Pow") and expr.is_Pow:
                try:
                    if expr.base == v and expr.exp == 2:
                        return -1.0
                except Exception:
                    pass
        except Exception:
            pass
        return 1.0

    @staticmethod
    def _extract_x_coefficient(expr, v) -> float:
        """Extract the coefficient c from a term like -c*x."""
        try:
            if hasattr(expr, "is_Mul") and expr.is_Mul:
                args_list = list(expr.args)
                coeff_parts = []
                for a in args_list:
                    if a == v:
                        continue
                    elif not a.has(v):
                        coeff_parts.append(a)
                if coeff_parts:
                    return float(Mul(*coeff_parts))
        except Exception:
            pass
        return -1.0

    def _probe_periodicity(self, expr, a: float, b: float) -> tuple:
        """
        Detect periodicity by scanning for sin/cos atoms and extracting their period.

        Fixed: uses only the atom-scan path (not endpoint value matching) to avoid
        double-counting. The previous approach could flag cos(3x) on [-pi, pi] as
        periodic via both the endpoint check AND the atom scan.
        """
        v = self._var
        period = None

        # Scan for sin/cos atoms and extract their angular frequency
        trig_atoms = list(expr.atoms(sin)) + list(expr.atoms(cos))
        if not trig_atoms:
            return False, None

        best_period = None
        for sub in trig_atoms:
            try:
                arg = sub.args[0]
                # Extract coefficient of v: arg should be c*v or similar
                coeff = arg.as_coeff_mul(v)[0] if hasattr(arg, "as_coeff_mul") else S.One
                cf = float(coeff)
                if math.isfinite(cf) and abs(cf) > 1e-12:
                    candidate_period = 2 * float(pi) / abs(cf)
                    # Use the smallest period (fundamental frequency)
                    if best_period is None or candidate_period < best_period:
                        best_period = candidate_period
            except Exception:
                pass

        if best_period is not None and (b - a) <= best_period + 1e-6:
            return True, best_period

        # Fallback: check endpoint matching for non-trig periodic functions
        try:
            fa = float(N(expr.subs(v, a), 10))
            fb = float(N(expr.subs(v, b), 10))
            if math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) < 1e-6:
                df = diff(expr, v)
                fa_prime = float(N(df.subs(v, a), 10)) if math.isfinite(fa) else None
                fb_prime = float(N(df.subs(v, b), 10)) if math.isfinite(fb) else None
                if (fa_prime is not None and fb_prime is not None
                        and abs(fa_prime - fb_prime) < 1e-4):
                    return True, b - a
        except Exception:
            pass

        return False, None

    # ------------------------------------------------------------------ #
    #  Recommendation engine                                              #
    # ------------------------------------------------------------------ #

    @dataclass
    class _Rec:
        family: PolynomialFamily
        confidence: str
        reason: str

    def _recommend_family(self, analysis: FunctionAnalysis, a: float, b: float) -> "QuadratureAnalyzer._Rec":
        iv = analysis.interval_type
        has_end_sg = analysis.has_endpoint_singularity
        has_int_sg = analysis.has_interior_singularity
        decay = analysis.decay_type
        is_periodic = analysis.is_periodic_on_interval

        if iv == "infinite":
            if decay in ("gaussian",):
                return self._Rec(PolynomialFamily.HERMITE, "high",
                                 "Infinite domain with Gaussian-weighted integrand. Gauss-Hermite quadrature absorbs the e^{-x^2} weight, yielding machine-precision results with O(sqrt(desired_digits)) nodes.")
            elif decay in ("exponential",):
                return self._Rec(PolynomialFamily.HERMITE, "medium",
                                 "Infinite domain with exponential decay. Gauss-Hermite can still be effective if the decay is fast enough; otherwise consider a variable transformation to a finite interval and use Legendre.")
            else:
                return self._Rec(PolynomialFamily.HERMITE, "low",
                                 "Infinite domain with unknown or slow decay. Gauss-Hermite may converge slowly; consider transforming the integral to a finite interval first.")

        if iv == "semi_infinite":
            if decay in ("exponential",):
                return self._Rec(PolynomialFamily.LAGUERRE, "high",
                                 "Semi-infinite domain with exponential decay. Gauss-Laguerre absorbs the e^{-x} weight, ideal for Laplace-type integrals.")
            elif decay in ("algebraic",):
                return self._Rec(PolynomialFamily.LAGUERRE, "medium",
                                 "Semi-infinite domain with algebraic decay. Gauss-Laguerre is usable but may need higher degree; consider variable substitution t = x/(1+x) to map [0, inf) -> [0, 1) and use Legendre.")
            else:
                return self._Rec(PolynomialFamily.LAGUERRE, "medium",
                                 "Semi-infinite domain. Gauss-Laguerre is the default choice; for slow decay consider mapping to a finite interval.")

        if iv == "finite":
            if has_end_sg:
                return self._Rec(PolynomialFamily.CHEBYSHEV, "high",
                                 "Endpoint singularity detected at " + str(analysis.singularities) + ". Chebyshev nodes cluster near endpoints, and Clenshaw-Curtis quadrature handles endpoint singularities much better than Gauss-Legendre.")
            if has_int_sg:
                return self._Rec(PolynomialFamily.LEGENDRE, "low",
                                 "Interior singularity detected at " + str(analysis.singularities) + ". Standard Gaussian quadrature will converge slowly. Consider splitting the integral at the singularity or applying a variable transformation to remove it.")
            if is_periodic:
                return self._Rec(PolynomialFamily.CHEBYSHEV, "high",
                                 "Function appears periodic on [" + str(a) + ", " + str(b) + "] with period ~" + str(analysis.approximate_period) + ". Clenshaw-Curtis quadrature exploits periodicity via FFT and achieves spectral accuracy.")
            return self._Rec(PolynomialFamily.LEGENDRE, "high",
                             "Smooth function on a finite interval with no singularities. Gauss-Legendre is the optimal choice: it integrates polynomials of degree up to 2n-1 exactly and converges exponentially for analytic integrands.")

        return self._Rec(PolynomialFamily.LEGENDRE, "low", "Could not determine optimal family; defaulting to Gauss-Legendre.")

    # ------------------------------------------------------------------ #
    #  Degree-selection criteria                                          #
    # ------------------------------------------------------------------ #

    def _degree_criteria(self, analysis: FunctionAnalysis) -> list:
        criteria: list = []
        fam = analysis.recommended_family
        iv = analysis.interval_type
        growth = analysis.derivative_growth_rate
        has_sg = analysis.has_endpoint_singularity or analysis.has_interior_singularity
        is_periodic = analysis.is_periodic_on_interval

        # General convergence principle
        criteria.append(
            "Gaussian quadrature with n nodes integrates exactly all polynomials of degree <= 2n-1. For non-polynomial integrands, the error decays as O(|f^{(2n)}(xi)|) for some xi in the interval.")

        # Smoothness-based criterion
        log_at_origin = any(s["kind"] == "log_zero" and abs(s["location"]) < 1e-10
                            for s in analysis.singularities
                            )
        if log_at_origin and iv == "semi_infinite":
            criteria.append("Log singularity at x=0 detected. Standard Gauss-Laguerre converges "
                            "algebraically for log-singular integrands (O(n^{-1/2}) typically). "
                            "For higher accuracy, consider Generalized Laguerre with alpha>0 to "
                            "pre-condition the singularity, or split: integrate over [0,1] with "
                            "a log-adapted rule and [1,inf) with Gauss-Laguerre separately."
                            )
        elif growth == "bounded":
            criteria.append("Derivatives are bounded on the interval => f is very smooth (likely "
                            "analytic). Exponential convergence is expected; n=16-32 usually "
                            "suffices for double-precision accuracy."
                            )
        elif growth == "polynomial":
            criteria.append(
                "Derivatives grow polynomially (like k!) => f is C^infty but not analytic. Algebraic convergence O(n^{-p}) expected; use n=32-128 and monitor convergence by doubling n.")
        elif growth == "exponential":
            criteria.append(
                "Derivatives grow exponentially (like C^k * k!) => f has a nearby complex singularity. Convergence is algebraic; use n=64-256 and consider a variable transformation to push singularities further away.")
        else:
            criteria.append(
                "Derivatives grow super-exponentially => f has a strong singularity or is not smooth. Standard Gaussian quadrature will converge slowly; consider splitting the interval, using a weight function that matches the singularity, or applying a change of variables.")

        # Singularity-based criterion
        if has_sg:
            # Suppress generic singularity message for log-at-origin on semi-infinite
            # intervals — the more specific message above already covers this case
            log_at_origin = any(
                s["kind"] == "log_zero" and abs(s["location"]) < 1e-10
                for s in analysis.singularities
            )
            if not (log_at_origin and iv == "semi_infinite"):
                criteria.append(
                    "Singularities present: " + str(analysis.singularities) + ". This degrades convergence from exponential to algebraic (O(n^{-alpha}) where alpha depends on the singularity strength). Use n=64-256 and prefer Chebyshev nodes which cluster near endpoints.")

        # Periodicity criterion
        if is_periodic:
            criteria.append(
                "Function is periodic on the interval (period ~" + str(analysis.approximate_period) + "). Clenshaw-Curtis quadrature achieves spectral accuracy; n=16-64 typically reaches machine precision.")

        # Infinite-interval criterion
        if iv == "infinite":
            criteria.append(
                "Infinite domain with decay type '" + analysis.decay_type + "'. The effective number of nodes needed scales as O(sqrt(-log(tol))) for Gaussian-weighted integrals. For double precision (tol ~ 1e-15), n=32-64 is usually sufficient with the matching weight function.")
        elif iv == "semi_infinite":
            criteria.append(
                "Semi-infinite domain with decay type '" + analysis.decay_type + "'. Gauss-Laguerre absorbs e^{-x}; for slower decays, n=64-128 may be needed.")

        # Family-specific criterion
        if fam == PolynomialFamily.LEGENDRE:
            criteria.append(
                "Gauss-Legendre: optimal for smooth integrands on finite intervals. Error bound ~ M_{2n} * (b-a)^{2n+1} / ((2n+1) * 4^n * (n!)^2), where M_{2n} = max|f^{(2n)}|.")
        elif fam == PolynomialFamily.CHEBYSHEV:
            criteria.append(
                "Gauss-Chebyshev / Clenshaw-Curtis: optimal when endpoint singularities are present or the function is periodic. Clenshaw-Curtis converges as O(n^{-r}) where r depends on smoothness; for analytic functions, convergence is exponential.")
        elif fam == PolynomialFamily.HERMITE:
            criteria.append(
                "Gauss-Hermite: optimal for integrals over (-inf, +inf) with e^{-x^2} weight. Error decays as O(M_{2n} / 4^n * n!). For f(x)=1 (pure Gaussian integral), n=8 already gives full double-precision accuracy.")
        else:
            criteria.append(
                "Gauss-Laguerre: optimal for [0, +inf) with e^{-x} weight. Error decays as O(M_{2n} / 4^n * n!). For f(x)=1 (pure Laguerre integral), n=8 gives full double-precision accuracy.")

        # Practical convergence check
        criteria.append(
            "Practical rule: compute the integral with n and 2n nodes; if |I_{2n} - I_n| < tol * (1 + |I_{2n}|), stop. Otherwise double n until convergence or until n exceeds the warning threshold (Hermite: n>=100, others: n>=200).")

        return criteria

    def _suggest_degree_range(self, analysis: FunctionAnalysis) -> tuple:
        growth = analysis.derivative_growth_rate
        has_sg = analysis.has_endpoint_singularity or analysis.has_interior_singularity
        iv = analysis.interval_type
        fam = analysis.recommended_family

        if growth == "bounded":
            lo, hi = 8, 32
        elif growth == "polynomial":
            lo, hi = 16, 64
        elif growth == "exponential":
            lo, hi = 32, 128
        else:
            lo, hi = 64, 256

        if has_sg:
            lo = max(lo, 32)
            hi = max(hi, 128)
        if iv == "infinite" and fam in (PolynomialFamily.HERMITE, PolynomialFamily.LAGUERRE):
            lo = min(lo, 16)
            hi = min(hi, 64)
        if analysis.is_periodic_on_interval:
            lo = max(8, lo // 2)
            hi = max(32, hi // 2)
        return (lo, hi)

    @staticmethod
    def _fmt_val(v) -> str:
        """Format a numeric value for code generation."""
        if v == float("inf"):
            # return "'inf'"
            return "float('inf')"
        if v == float("-inf"):
            # return "'-inf'"
            return "float('-inf')"
        return f"{v:.6g}"

    def recommend_usage(self, analysis: FunctionAnalysis) -> str:
        """
        Return a ready-to-copy Python snippet using the recommended family.

        Fixed: uses interval_a/interval_b from the analysis (actual integration bounds)
        instead of singularity locations. The previous version used singularities[0][location]
        as the lower bound, which is wrong for smooth functions and ignored the actual interval.
        """
        fam = analysis.recommended_family
        n_max = analysis.suggested_max_n
        var = analysis.variable
        a_val = self._fmt_val(analysis.interval_a)
        b_val = self._fmt_val(analysis.interval_b)

        if fam == PolynomialFamily.LEGENDRE:
            return (
                f"# Recommended: Gauss-Legendre on [{analysis.original_expr}]\n"
                f"from legendre import LegendreQuadrature\n"
                f"import numpy as np\n\n"
                f"quad = LegendreQuadrature(n={n_max}, use_mpmath=False)\n"
                f"result = quad.integrate_transformed(\n"
                f"    lambda {var}: {analysis.original_expr},\n"
                f"    a={a_val}, b={b_val},\n"
                f"    nodes=quad.nodes, weights=quad.weights\n"
                f")\n"
            )
        elif fam == PolynomialFamily.CHEBYSHEV:
            return (
                f"# Recommended: Gauss-Chebyshev or Clenshaw-Curtis\n"
                f"from chebyshev import ChebyshevQuadrature\n"
                f"import numpy as np\n\n"
                f"q = ChebyshevQuadrature()\n"
                f"result = q.gauss_chebyshev_quadrature(\n"
                f"    lambda {var}: {analysis.original_expr}, n={n_max}\n"
                f")\n"
            )
        elif fam == PolynomialFamily.HERMITE:
            return (
                f"# Recommended: Gauss-Hermite on (-inf, +inf)\n"
                f"from hermite import GaussHermiteQuadrature\n"
                f"import numpy as np\n\n"
                f"quad = GaussHermiteQuadrature(n={n_max}, use_mpmath=True)\n"
                f"result = quad.integrate(lambda {var}: {analysis.original_expr})\n"
            )
        else:
            return (
                f"# Recommended: Gauss-Laguerre on [0, +inf)\n"
                f"from laguerre import LaguerreQuadrature\n"
                f"import numpy as np\n\n"
                f"quad = LaguerreQuadrature(n={n_max}, alpha=0.0, use_mpmath=False)\n"
                f"result = quad.integrate(lambda {var}: {analysis.original_expr})\n"
            )

    def execute_quadrature(
        self,
        expression,
        interval=None,
        *,
        variable: str = "x",
        n: Optional[int] = None,
        tol: float = 1e-12,
        use_mpmath: bool = False,
    ) -> QuadratureResult:
        """
        Actually compute the definite integral of expression over interval
        using the recommended orthogonal-polynomial family from OrthoPolyB_np_mp.

        Parameters
        ----------
        expression : str | sympy.Expr
            The mathematical expression defining f(variable).
        interval : tuple[float, float] | None
            (a, b) for the integration domain. If None, inferred automatically.
        variable : str
            Name of the independent variable.
        n : int | None
            Number of quadrature nodes. If None, uses suggested_max_n from analysis.
        tol : float
            Tolerance for adaptive degree doubling.
        use_mpmath : bool
            Use mpmath arbitrary-precision arithmetic (default False).

        Returns
        -------
        QuadratureResult with value, family_used, n_nodes, converged, error_estimate.
        """
        # Step 1: analyse the function
        analysis = self.analyze(expression, interval=interval, variable=variable)
        fam = analysis.recommended_family

        # Determine number of nodes
        if n is None:
            n = analysis.suggested_max_n

        # Step 2: convert SymPy expression to a callable numpy function
        v = Symbol(variable)
        expr = self._parse(expression, variable)
        try:
            func = self._sympy_to_numpy(expr, variable)
        except Exception as e:
            return QuadratureResult(
                value=float("nan"), family_used=fam, n_nodes=n,
                converged=False, error_estimate=None,
                message="Failed to convert expression to numpy callable: " + str(e),
            )

        # BEGIN OP46
        try:
            if fam == PolynomialFamily.LEGENDRE:
                value = self._integrate_legendre(func, analysis.interval_a, analysis.interval_b, n)
            elif fam == PolynomialFamily.CHEBYSHEV:
                '''
                value = self._integrate_chebyshev(func, analysis.interval_a, analysis.interval_b, n,
                                                  has_endpoint_singularity=analysis.has_endpoint_singularity)
                '''
                value = self._integrate_chebyshev(func, expr, variable,
                                                  analysis.interval_a, analysis.interval_b, n,
                                                  has_endpoint_singularity=analysis.has_endpoint_singularity)
            elif fam == PolynomialFamily.HERMITE:
                value = self._integrate_hermite(expr, variable, n, use_mpmath)
            else:  # LAGUERRE
                value = self._integrate_laguerre(expr, variable, n)
        except ImportError as e:
            return QuadratureResult(
                value=float("nan"), family_used=fam, n_nodes=n,
                converged=False, error_estimate=None,
                message="Cannot import OrthoPolyB_np_mp module: " + str(e) + ". Make sure the quadrature modules are in your PYTHONPATH.",
            )
        except Exception as e:
            return QuadratureResult(
                value=float("nan"), family_used=fam, n_nodes=n,
                converged=False, error_estimate=None,
                message="Quadrature computation failed: " + str(e),
            )
        # END OP46

        # Step 4: adaptive degree doubling for convergence check
        value_n = float(value)
        if not math.isfinite(value_n):
            return QuadratureResult(
                value=value_n, family_used=fam, n_nodes=n,
                converged=False, error_estimate=None,
                message="Initial quadrature returned NaN at n=" + str(n),
            )

        n2 = min(n * 2, 200)  # cap at warning threshold

        # BEGIN OP46
        try:
            if fam == PolynomialFamily.LEGENDRE:
                value_2n = self._integrate_legendre(func, analysis.interval_a, analysis.interval_b, n2)
            elif fam == PolynomialFamily.CHEBYSHEV:
                value_2n = self._integrate_chebyshev(func, expr, variable,
                                                     analysis.interval_a, analysis.interval_b, n2,
                                                     has_endpoint_singularity=analysis.has_endpoint_singularity)
            elif fam == PolynomialFamily.HERMITE:
                value_2n = self._integrate_hermite(expr, variable, n2, use_mpmath)
            else:
                value_2n = self._integrate_laguerre(expr, variable, n2)
        except Exception as e:
            print(f"DEBUG n2 failed: {e}")
            return QuadratureResult(
                value=value_n, family_used=fam, n_nodes=n,
                converged=False, error_estimate=None,
                message="Convergence check failed at n=" + str(n2),
            )
        # END OP46

        value_2n = float(value_2n)
        if not math.isfinite(value_2n):
            return QuadratureResult(
                value=value_n, family_used=fam, n_nodes=n,
                converged=False, error_estimate=None,
                message="Quadrature at 2n returned NaN",
            )

        err = abs(value_2n - value_n)
        rel_err = err / (abs(value_n) + 1e-30)
        # converged = rel_err < tol
        converged = (err < tol) or (rel_err < tol)
        print(f"DEBUG: n={n}, n2={n2}, err={err:.3e}, rel_err={rel_err:.3e}, tol={tol}")
        return QuadratureResult(
            value=value_2n if converged else value_n,
            family_used=fam,
            n_nodes=n2 if converged else n,
            converged=converged,
            error_estimate=err,
            message="Converged" if converged else "Did not converge within tol=" + str(tol),
        )

    # ------------------------------------------------------------------ #
    #  Actual quadrature execution helpers                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sympy_to_numpy(expr, variable: str):
        """
        Convert a SymPy expression to a numpy-compatible callable.

        Uses lambdify for automatic conversion of sympy functions (sin, cos, exp,
        log, sqrt) to their numpy equivalents.
        """
        from sympy import lambdify
        v = Symbol(variable)
        return lambdify(v, expr, modules="numpy")

    def _integrate_legendre(self, func, a: float, b: float, n: int) -> float:
        """Integrate using Gauss-Legendre via OrthoPolyB_np_mp. Returns the integral value."""
        from legendre import LegendreQuadrature
        quad = LegendreQuadrature(n=n, use_mpmath=False)
        result = quad.integrate_transformed(func, 
                                            a=a, 
                                            b=b,
                                            nodes=quad.nodes, 
                                            weights=quad.weights,
                                            )
        return float(result)

    # BEGIN OP46
    def _integrate_chebyshev(self, func, expr, variable, a, b, n,
                             has_endpoint_singularity=False):
        from chebyshev import clencurt_quadrature, ChebyshevQuadrature
        if has_endpoint_singularity:
            from sympy import sqrt, Symbol, lambdify
            v = Symbol(variable)
            weight = 1 / sqrt(1 - v**2)
            stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")
            q = ChebyshevQuadrature()
            return float(q.gauss_chebyshev_quadrature(stripped, n=n))
        else:
            scale = (b - a) / 2.0
            shift = (b + a) / 2.0
            mapped = lambda x: func(scale * x + shift)
            return float(clencurt_quadrature(mapped, n) * scale)

    def _integrate_hermite(self, expr, variable: str, n: int,
                           use_mpmath: bool = False) -> float:
        from sympy import exp, Symbol, lambdify
        from hermite import GaussHermiteQuadrature
        v = Symbol(variable)
        weight = exp(-v**2)
        stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")

        # Sanity check: if stripped function grows at large |x|, Hermite will diverge
        test_vals = stripped(np.array([3.0, 5.0, 7.0]))
        if np.any(np.abs(test_vals) > 1e6):
            raise ValueError(
                "Stripped integrand grows at large |x| — Gauss-Hermite will diverge. "
                "Consider transforming to a finite interval and using Legendre."
            )

        quad = GaussHermiteQuadrature(n=n, use_mpmath=use_mpmath)
        return float(quad.integrate(stripped))

    def _integrate_laguerre(self, expr, variable, n):
        from sympy import exp, Symbol, lambdify
        v = Symbol(variable)
        weight = exp(-v)
        stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")
        from laguerre import gauss_quadrature_weights
        nodes, weights = gauss_quadrature_weights(n)
        return float(np.sum(weights * stripped(nodes)))
    # END OP46

# ---------------------------------------------------------------------------
#  Demo / self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    analyzer = QuadratureAnalyzer()

    examples = [
        ("exp(-x**2) * sin(x)", (-1, 1), "Smooth analytic function on [-1,1]"),
        ("1 / sqrt(1 - x**2)", (-1, 1), "Endpoint singularity at +/-1"),
        ("exp(-x**2)", None, "Gaussian decay => infinite interval"),
        ("log(x) * exp(-x)", None, "Log singularity + exponential decay"),
        ("cos(3*x)", (-float(pi), float(pi)), "Periodic function on [-pi, pi]"),
        ("x**2 * exp(-x)", (0, float("inf")), "Laguerre-type integral"),
        ("1 / (1 + x**4)", (-float("inf"), float("inf")), "Algebraic decay on infinite interval"),
    ]

    for expr_str, interval, desc in examples:
        print(f"\n{'=' * 70}")
        print(f"  {desc}")
        print(f"  Expression: {expr_str}")
        print(f"  Interval:   {interval}")
        print(f"{'=' * 70}")

        analysis = analyzer.analyze(expr_str, interval=interval)

        print(f"\n  Recommended family : {analysis.recommended_family.value}")
        print(f"  Confidence         : {analysis.confidence}")
        reason_preview = analysis.recommendation_reason[:120]
        print(f"  Reason             : {reason_preview}...")
        print(f"  Suggested degree   : [{analysis.suggested_min_n}, {analysis.suggested_max_n}]")

        if analysis.singularities:
            print(f"  Singularities      : {analysis.singularities}")
        if analysis.is_periodic_on_interval:
            print(f"  Periodic           : True (period ~{analysis.approximate_period})")
        print(f"  Derivative growth  : {analysis.derivative_growth_rate}")

        # Show actual quadrature computation
        try:
            result = analyzer.execute_quadrature(expr_str, interval=interval)
            print(f"\n  Computed integral  : {result.value:.15e}")
            print(f"  Family used        : {result.family_used.value}")
            print(f"  Nodes              : {result.n_nodes}")
            print(f"  Converged          : {result.converged}")
            if result.error_estimate is not None:
                print(f"  Error estimate     : {result.error_estimate:.3e}")
        except Exception as e:
            print(f"\n  Quadrature error   : {e}")

        print("\n  Degree-selection criteria:")
        for i, c in enumerate(analysis.degree_criteria, 1):
            words = c.split()
            line = ""
            for w in words:
                if len(line) + len(w) + 1 > 76:
                    print(f"    {line}")
                    line = f"      {w}"
                else:
                    prefix = "" if not line else " "
                    line = f"{line}{prefix}{w}"
            if line:
                print(f"    {line}")

        print("\n  Usage snippet:")
        usage = analyzer.recommend_usage(analysis)
        for uline in usage.strip().splitlines():
            print(f"    {uline}")

    print(f"\n{'=' * 70}")
    print("  Done.")
