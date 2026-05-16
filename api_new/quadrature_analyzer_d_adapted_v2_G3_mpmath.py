# quadrature_analyzer_d_adapted_v2_G3_mpmath.py
# MPMATH CAPABILITY ADDITION — Full mpmath pipeline for ALL polynomial families.
#   - A1: Added mpmath import at module level
#   - A2: _integrate_legendre now accepts use_mpmath parameter; calls LegendreQuadrature.integrate_mp()
#   - A3: _integrate_chebyshev now accepts use_mpmath parameter; uses ClenshawCurtisMP
#   - A4: _integrate_hermite full mpmath pipeline with integrate_mp, returns mp.mpf
#   - A5: _integrate_laguerre now accepts use_mpmath parameter; calls LaguerreQuadrature.integrate_mp()
#   - A6: _integrate_jacobi and _integrate_with_jacobi_weight accept use_mpmath; use mpmath.quad when True
#   - A7: execute_quadrature dispatch propagates use_mpmath to ALL integration methods
#   - A8: Convergence checking keeps mp.mpf values until final QuadratureResult construction
from __future__ import annotations
import os
import sys
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

# A1: Add mpmath import at module level
import mpmath as mp
from sympy import (
    Symbol, diff, N, S, pi, log, sin, cos, tan, exp, sqrt,
    Abs as SymAbs, limit, solve, Derivative, Mul, Pow, lambdify,
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
    hermite_compatible: bool = True
    is_periodic_on_interval: bool = False
    approximate_period: Optional[float] = None
    recommended_family: PolynomialFamily = PolynomialFamily.LEGENDRE
    confidence: str = "high"
    recommendation_reason: str = ""
    suggested_min_n: int = 8
    suggested_max_n: int = 64
    degree_criteria: list[str] = field(default_factory=list)
    interval_a: float = -1.0
    interval_b: float = 1.0
    # CATEGORY C: endpoint singularity exponents for Jacobi weight matching
    left_singularity_alpha: float = 0.0   # exponent of (x-a)^(-alpha) at left endpoint
    right_singularity_beta: float = 0.0   # exponent of (b-x)^(-beta) at right endpoint
    has_algebraic_endpoint_singularity: bool = False


def _safe_eval_sympy(expr, x_val):
    try:
        return float(N(expr.subs("x", x_val), 15))
    except Exception:
        return float("nan")


class QuadratureAnalyzer:

    def __init__(self, default_variable: str = "x"):
        self._var = Symbol(default_variable)

    # ------------------------------------------------------------------ #
    #  analyze() — unchanged from original                               #
    # ------------------------------------------------------------------ #

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
            decay_type, decay_rate, hermite_compat = self._probe_decay(expr, variable)
            analysis.decay_type = decay_type
            analysis.decay_rate = decay_rate
            analysis.hermite_compatible = hermite_compat
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

        # CATEGORY C FIX: extract endpoint singularity exponents for Jacobi weight matching
        if not np.isinf(a) and not np.isinf(b):
            exp_data = self._extract_endpoint_singularity_exponents(expr, a, b)
            analysis.left_singularity_alpha = exp_data.get("left", 0.0)
            analysis.right_singularity_beta = exp_data.get("right", 0.0)
            analysis.has_algebraic_endpoint_singularity = (
                analysis.left_singularity_alpha > 0 or analysis.right_singularity_beta > 0
            )

        return analysis

    # ------------------------------------------------------------------ #
    #  Internal helpers (unchanged from original)                        #
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
        v = Symbol(variable)
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_quadratic(inner, v):
                    return (-float("inf"), float("inf"))
            except Exception:
                pass
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_linear(inner, v):
                    return (0.0, float("inf"))
            except Exception:
                pass
        for sub in expr.atoms(log):
            try:
                arg = sub.args[0]
                if arg == v or (hasattr(arg, "is_Mul") and arg.is_Mul and len(arg.args) == 1 and arg.args[0] == v):
                    return (0.0, float("inf"))
            except Exception:
                pass
        try:
            den = expr.as_numer_denom()[1]
            if den.has(v):
                for pt in den.atoms():
                    if hasattr(pt, "is_Pow") and pt.is_Pow and pt.exp < 0:
                        if pt.base == v or (hasattr(pt.base, "is_Mul") and pt.base.is_Mul):
                            return (0.0, float("inf"))
        except Exception:
            pass
        return (-1.0, 1.0)

    @staticmethod
    def _is_negative_quadratic(expr, v) -> bool:
        try:
            expanded = expr.expand()
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
            neg_v2 = -v ** 2
            if expr == neg_v2:
                return True
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
            if hasattr(expr, "is_Pow") and expr.is_Pow:
                try:
                    if expr.base == v and expr.exp == 2:
                        return False
                except Exception:
                    pass
        except Exception:
            pass
        return False

    @staticmethod
    def _is_negative_linear(expr, v) -> bool:
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
        v = self._var
        if np.isinf(a) or np.isinf(b):
            probe_points = [0.0]
            for k in range(1, 6):
                probe_points.append(float(k))
                probe_points.append(-float(k))
        else:
            n_probe = min(11, max_order + 2)
            try:
                raw_nodes, _ = np.polynomial.legendre.leggauss(n_probe)
                scale = (b - a) / 2.0
                shift = (b + a) / 2.0
                probe_points = [float(scale * n_val + shift) for n_val in raw_nodes]
            except ImportError:
                probe_points = np.linspace(a, b, min(11, max_order + 2)).tolist()

        all_ratios: list[float] = []

        for pt in probe_points:
            deriv_magnitudes: list[float] = []
            for k in range(1, min(max_order + 1, 7)):
                dexpr = diff(expr, v, k)
                try:
                    val = float(N(dexpr.subs(v, pt), 10))
                    mag = abs(val)
                    if math.isfinite(mag):
                        deriv_magnitudes.append(mag)
                except Exception:
                    break

            if len(deriv_magnitudes) >= 2:
                for i in range(1, len(deriv_magnitudes)):
                    prev = deriv_magnitudes[i - 1]
                    curr = deriv_magnitudes[i]
                    if prev < 1e-30 or curr < 1e-30:
                        continue
                    log_prev = math.log(prev)
                    if abs(log_prev) < 1e-15:
                        continue
                    ratio = math.log(curr) / log_prev
                    all_ratios.append(ratio)

        if not all_ratios:
            return "bounded"

        avg_ratio = np.mean(all_ratios)
        if avg_ratio < 1.5:
            return "bounded"
        elif avg_ratio < 2.5:
            return "polynomial"
        elif avg_ratio < 6.0:
            return "exponential"
        else:
            return "super_exponential"

    def _probe_decay(self, expr, variable: str) -> tuple:
        v = Symbol(variable)
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_quadratic(inner, v):
                    expanded = inner.expand()
                    coeff = self._extract_x2_coefficient(expanded, v)
                    c = abs(coeff)
                    hermite_ok = c >= 1.0 - 1e-12
                    return ("gaussian", c, hermite_ok)
            except Exception:
                pass
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_linear(inner, v):
                    coeff = self._extract_x_coefficient(inner, v)
                    return ("exponential", abs(coeff), False)
            except Exception:
                pass
        try:
            den = expr.as_numer_denom()[1]
            if den.has(v):
                for pt in den.atoms():
                    if hasattr(pt, "is_Pow") and pt.is_Pow and pt.exp < 0:
                        return ("algebraic", float(-pt.exp), False)
        except Exception:
            pass
        if expr.has(sin) or expr.has(cos):
            return ("oscillatory", 0.0, False)
        return ("none", 0.0, False)

    @staticmethod
    def _extract_x2_coefficient(expr, v) -> float:
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
        v = self._var
        trig_atoms = list(expr.atoms(sin)) + list(expr.atoms(cos))
        if not trig_atoms:
            return False, None

        best_period = None
        for sub in trig_atoms:
            try:
                arg = sub.args[0]
                coeff = arg.as_coeff_mul(v)[0] if hasattr(arg, "as_coeff_mul") else S.One
                cf = float(coeff)
                if math.isfinite(cf) and abs(cf) > 1e-12:
                    candidate_period = 2 * float(pi) / abs(cf)
                    if best_period is None or candidate_period < best_period:
                        best_period = candidate_period
            except Exception:
                pass

        if best_period is not None and (b - a) <= best_period + 1e-6:
            return True, best_period

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
                if analysis.hermite_compatible:
                    return self._Rec(PolynomialFamily.HERMITE, "high",
                                     "Infinite domain with Gaussian-weighted integrand.")
                else:
                    return self._Rec(PolynomialFamily.LEGENDRE, "medium",
                                     f"Gaussian decay rate c={analysis.decay_rate:.4f} < 1.0; recommend finite-window Legendre.")
            elif decay in ("exponential",):
                return self._Rec(PolynomialFamily.HERMITE, "medium",
                                 "Infinite domain with exponential decay.")
            else:
                return self._Rec(PolynomialFamily.HERMITE, "low",
                                 "Infinite domain with unknown or slow decay.")

        if iv == "semi_infinite":
            if decay in ("exponential",):
                return self._Rec(PolynomialFamily.LAGUERRE, "high",
                                 "Semi-infinite domain with exponential decay.")
            elif decay in ("algebraic",):
                return self._Rec(PolynomialFamily.LAGUERRE, "medium",
                                 "Semi-infinite domain with algebraic decay.")
            else:
                return self._Rec(PolynomialFamily.LAGUERRE, "medium",
                                 "Semi-infinite domain. Gauss-Laguerre is the default choice.")

        if iv == "finite":
            if has_end_sg:
                return self._Rec(PolynomialFamily.CHEBYSHEV, "high",
                                 "Endpoint singularity detected.")
            if has_int_sg:
                return self._Rec(PolynomialFamily.LEGENDRE, "low",
                                 "Interior singularity detected.")
            if is_periodic:
                return self._Rec(PolynomialFamily.CHEBYSHEV, "high",
                                 "Function appears periodic on the interval.")
            return self._Rec(PolynomialFamily.LEGENDRE, "high",
                             "Smooth function on a finite interval with no singularities.")

        return self._Rec(PolynomialFamily.LEGENDRE, "low", "Could not determine optimal family.")

    # ------------------------------------------------------------------ #
    #  Degree-selection criteria                                          #
    # ------------------------------------------------------------------ #

    def _degree_criteria(self, analysis: FunctionAnalysis) -> list:
        criteria: list = []
        growth = analysis.derivative_growth_rate
        has_sg = analysis.has_endpoint_singularity or analysis.has_interior_singularity
        is_periodic = analysis.is_periodic_on_interval
        iv = analysis.interval_type

        criteria.append(
            "Gaussian quadrature with n nodes integrates exactly all polynomials of degree <= 2n-1.")

        log_at_origin = any(s["kind"] == "log_zero" and abs(s["location"]) < 1e-10
                            for s in analysis.singularities)
        if log_at_origin and iv == "semi_infinite":
            criteria.append("Log singularity at x=0 detected.")
        elif growth == "bounded":
            criteria.append("Derivatives are bounded => exponential convergence expected; n=16-32 usually suffices.")
        elif growth == "polynomial":
            criteria.append("Derivatives grow polynomially => algebraic convergence O(n^{-p}); use n=32-128.")
        elif growth == "exponential":
            criteria.append("Derivatives grow exponentially => algebraic convergence; use n=64-256.")
        else:
            criteria.append("Derivatives grow super-exponentially => consider splitting the interval.")

        if has_sg and not (log_at_origin and iv == "semi_infinite"):
            criteria.append("Singularities present: " + str(analysis.singularities))

        if is_periodic:
            criteria.append("Function is periodic on the interval.")

        if iv == "infinite":
            criteria.append("Infinite domain with decay type '" + analysis.decay_type + "'.")
        elif iv == "semi_infinite":
            criteria.append("Semi-infinite domain with decay type '" + analysis.decay_type + "'.")

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

        # CATEGORY B FIX: inflate hi based on oscillation frequency.
        if not (np.isinf(analysis.interval_a) or np.isinf(analysis.interval_b)):
            omega_max = self._estimate_max_oscillation_frequency(
                analysis.sympy_expr, analysis.variable,
                analysis.interval_a, analysis.interval_b
            )
            if omega_max > 1e-6:
                interval_width = abs(analysis.interval_b - analysis.interval_a)
                n_half_periods = omega_max * interval_width / math.pi
                min_osc_nodes = int(10 * n_half_periods)
                hi = max(hi, min(min_osc_nodes, 500))

        return (lo, hi)

    @staticmethod
    def _fmt_val(v) -> str:
        if v == float("inf"):
            return "float('inf')"
        if v == float("-inf"):
            return "float('-inf')"
        return f"{v:.6g}"

    def recommend_usage(self, analysis: FunctionAnalysis) -> str:
        fam = analysis.recommended_family
        n_max = analysis.suggested_max_n
        var = analysis.variable
        a_val = self._fmt_val(analysis.interval_a)
        b_val = self._fmt_val(analysis.interval_b)

        if fam == PolynomialFamily.LEGENDRE:
            return (f"# Recommended: Gauss-Legendre on [{analysis.original_expr}]\n"
                    f"from legendre import LegendreQuadrature\n"
                    f"a, b = {a_val}, {b_val}\n"
                    f"quad = LegendreQuadrature(n={n_max}, use_mpmath=False)\n")
        elif fam == PolynomialFamily.CHEBYSHEV:
            return (f"# Recommended: Clenshaw-Curtis on [{analysis.original_expr}]\n"
                    f"from chebyshev import clencurt_integrate_interval\n"
                    f"a, b = {a_val}, {b_val}\n")
        elif fam == PolynomialFamily.HERMITE:
            return (f"# Recommended: Gauss-Hermite on (-inf, +inf)\n"
                    f"from hermite import GaussHermiteQuadrature\n"
                    f"quad = GaussHermiteQuadrature(n={n_max}, use_mpmath=True)\n")
        else:
            return (f"# Recommended: Gauss-Laguerre on [0, +inf)\n"
                    f"from laguerre import LaguerreQuadrature\n"
                    f"quad = LaguerreQuadrature(n={n_max}, alpha=0.0, use_mpmath=False)\n")

    # ------------------------------------------------------------------ #
    #  A7+A8: execute_quadrature with full mpmath propagation            #
    # ------------------------------------------------------------------ #

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
        analysis = self.analyze(expression, interval=interval, variable=variable)
        expr = analysis.sympy_expr
        fam = analysis.recommended_family
        v = Symbol(variable)

        if n is None:
            n = analysis.suggested_max_n

        current_a, current_b = analysis.interval_a, analysis.interval_b

        # 1. Singular Endpoint Detection (Category C)
        left_alpha, right_beta = 0.0, 0.0
        use_jacobi = False
        if analysis.interval_type == "finite" and analysis.has_endpoint_singularity:
            exp_data = self._extract_endpoint_singularity_exponents(expr, current_a, current_b)
            left_alpha = exp_data.get("left", 0.0)
            right_beta = exp_data.get("right", 0.0)
            use_jacobi = (left_alpha > 0 or right_beta > 0)

        # 2. Infinite Windowing (G3 Fallback)
        if np.isinf(current_a) or np.isinf(current_b):
            if fam == PolynomialFamily.LEGENDRE:
                L = self._compute_effective_support(expr, variable)
                current_a = -L if np.isinf(current_a) else current_a
                current_b = L if np.isinf(current_b) else current_b
                print(f"DEBUG: Infinite Legendre fallback window: [{current_a}, {current_b}]")

        # 3. Oscillation Bump
        if not (np.isinf(current_a) or np.isinf(current_b)):
            osc_safe_n = self._compute_oscillation_safe_n(expr, variable, current_a, current_b, n)
            if osc_safe_n > n:
                print(f"DEBUG: Oscillation bump n={n} -> {osc_safe_n}")
                n = osc_safe_n

        # 4. Create numpy callable for standard paths
        try:
            from sympy import lambdify
            func = lambdify(v, expr, modules="numpy")
        except Exception as e:
            return QuadratureResult(value=float("nan"), family_used=fam, n_nodes=n, converged=False, message=f"Lambdify failed: {e}")

        # A7: Execute integration — propagate use_mpmath to ALL families
        try:
            if use_jacobi:
                print(f"DEBUG: Using Gauss-Jacobi (a={left_alpha:.3f}, b={right_beta:.3f})")
                value = self._integrate_with_jacobi_weight(expr, variable, n, current_a, current_b, left_alpha, right_beta, use_mpmath=use_mpmath)
            elif fam == PolynomialFamily.LEGENDRE:
                if use_mpmath:
                    value = self._integrate_legendre(expr, current_a, current_b, n, use_mpmath=True)
                else:
                    value = self._integrate_legendre(func, current_a, current_b, n, use_mpmath=False)
            elif fam == PolynomialFamily.CHEBYSHEV:
                if use_mpmath:
                    value = self._integrate_chebyshev(expr, expr, variable, current_a, current_b, n,
                                                      has_endpoint_singularity=analysis.has_endpoint_singularity, use_mpmath=True)
                else:
                    value = self._integrate_chebyshev(func, expr, variable, current_a, current_b, n,
                                                      has_endpoint_singularity=analysis.has_endpoint_singularity, use_mpmath=False)
            elif fam == PolynomialFamily.HERMITE:
                value = self._integrate_hermite(expr, variable, n, use_mpmath)
            else:  # LAGUERRE
                value = self._integrate_laguerre(expr, variable, n, use_mpmath=use_mpmath)
        except Exception as e:
            return QuadratureResult(value=float("nan"), family_used=fam, n_nodes=n, converged=False, message=str(e))

        # A8: Convergence check — keep mp.mpf when use_mpmath=True
        if use_mpmath:
            value_n_mp = value  # keep as mp.mpf
            n2 = min(n * 2, 150)  # cap for mpmath (Golub-Welsch slow at high N)

            try:
                if use_jacobi:
                    value_2n_mp = self._integrate_with_jacobi_weight(expr, variable, n2, current_a, current_b, left_alpha, right_beta, use_mpmath=True)
                elif fam == PolynomialFamily.LEGENDRE:
                    value_2n_mp = self._integrate_legendre(expr, current_a, current_b, n2, use_mpmath=True)
                elif fam == PolynomialFamily.CHEBYSHEV:
                    value_2n_mp = self._integrate_chebyshev(expr, expr, variable, current_a, current_b, n2,
                                                            has_endpoint_singularity=analysis.has_endpoint_singularity, use_mpmath=True)
                elif fam == PolynomialFamily.HERMITE:
                    value_2n_mp = self._integrate_hermite(expr, variable, n2, use_mpmath=True)
                else:
                    value_2n_mp = self._integrate_laguerre(expr, variable, n2, use_mpmath=True)

                # Compare in mpf precision
                err_mp = abs(value_2n_mp - value_n_mp)
                rel_err_mp = err_mp / (abs(value_n_mp) + mp.mpf('1e-30'))
                tol_mp = mp.mpf(str(tol))
                converged = bool(err_mp < tol_mp or rel_err_mp < tol_mp)

                return QuadratureResult(
                    value=float(value_2n_mp if converged else value_n_mp),
                    family_used=fam,
                    n_nodes=n2 if converged else n,
                    converged=converged,
                    error_estimate=float(err_mp),
                    message="Converged" if converged else f"Did not meet tol (err={float(err_mp):.2e})"
                )
            except Exception as e:
                return QuadratureResult(value=float(value_n_mp), family_used=fam, n_nodes=n, converged=False, message=f"Check failed: {e}")
        else:
            # Standard float path (unchanged behavior)
            value_n = float(value)
            n2 = min(n * 2, 1000)

            try:
                if use_jacobi:
                    value_2n = self._integrate_with_jacobi_weight(expr, variable, n2, current_a, current_b, left_alpha, right_beta, use_mpmath=False)
                elif fam == PolynomialFamily.LEGENDRE:
                    value_2n = self._integrate_legendre(func, current_a, current_b, n2, use_mpmath=False)
                elif fam == PolynomialFamily.CHEBYSHEV:
                    value_2n = self._integrate_chebyshev(func, expr, variable, current_a, current_b, n2,
                                                         has_endpoint_singularity=analysis.has_endpoint_singularity, use_mpmath=False)
                elif fam == PolynomialFamily.HERMITE:
                    value_2n = self._integrate_hermite(expr, variable, n2, use_mpmath=False)
                else:
                    value_2n = self._integrate_laguerre(expr, variable, n2, use_mpmath=False)

                value_2n = float(value_2n)
                err = abs(value_2n - value_n)
                rel_err = err / (abs(value_n) + 1e-30)
                converged = (err < tol) or (rel_err < tol)

                return QuadratureResult(
                    value=value_2n if converged else value_n,
                    family_used=fam,
                    n_nodes=n2 if converged else n,
                    converged=converged,
                    error_estimate=err,
                    message="Converged" if converged else f"Did not meet tol (err={err:.2e})"
                )
            except Exception as e:
                return QuadratureResult(value=value_n, family_used=fam, n_nodes=n, converged=False, message=f"Check failed: {e}")

    # ------------------------------------------------------------------ #
    #  Actual quadrature execution helpers                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sympy_to_numpy(expr, variable: str):
        from sympy import lambdify
        v = Symbol(variable)
        return lambdify(v, expr, modules="numpy")

    # A2: Modified _integrate_legendre with use_mpmath parameter
    def _integrate_legendre(self, func_or_expr, a: float, b: float, n: int,
                             use_mpmath: bool = False):
        """Integrate on [a,b] using Gauss-Legendre.

        If use_mpmath=True, func_or_expr should be a sympy expression (not numpy callable).
        Returns float when use_mpmath=False, mp.mpf when True.
        """
        from legendre import LegendreQuadrature

        if use_mpmath:
            quad = LegendreQuadrature(n=n, use_mpmath=True, dps=80)
            return quad.integrate_mp(func_or_expr, a=a, b=b)  # returns mp.mpf
        else:
            scale = (b - a) / 2.0
            shift = (b + a) / 2.0
            quad = LegendreQuadrature(n=n, use_mpmath=False)
            transformed = lambda t: func_or_expr(scale * t + shift)
            result = quad.integrate(transformed)
            return float(result) * scale

    # A3: Modified _integrate_chebyshev with use_mpmath parameter
    def _integrate_chebyshev(self, func_or_expr, expr, variable, a, b, n,
                             has_endpoint_singularity=False, use_mpmath=False):
        if use_mpmath:
            from chebyshev.integration_mp import ClenshawCurtisMP
            cc = ClenshawCurtisMP(n, dps=80)
            v = Symbol(variable)
            f_mp = lambdify(v, expr, modules='mpmath')
            return cc.integrate_on_interval(f_mp, a, b)  # returns mp.mpf
        else:
            from chebyshev import clencurt_integrate_interval, ChebyshevQuadrature
            if has_endpoint_singularity:
                v = Symbol(variable)
                weight = 1 / sqrt(1 - v**2)
                stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")
                q = ChebyshevQuadrature()
                return float(q.clenshaw_curtis_quadrature(stripped, n=n))
            else:
                return float(clencurt_integrate_interval(func_or_expr, a, b, n))

    def _compute_effective_support(self, expr, variable: str, epsilon: float = 1e-15) -> float:
        v = Symbol(variable)
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_quadratic(inner, v):
                    coeff = abs(self._extract_x2_coefficient(inner.expand(), v))
                    return math.sqrt(math.log(1.0 / epsilon) / max(coeff, 1e-15))
                elif QuadratureAnalyzer._is_negative_linear(inner, v):
                    coeff = abs(self._extract_x_coefficient(inner, v))
                    return math.log(1.0 / epsilon) / max(coeff, 1e-15)
            except Exception:
                pass
        return 20.0

    # A4: Modified _integrate_hermite with full mpmath pipeline
    def _integrate_hermite(self, expr, variable: str, n: int,
                           use_mpmath: bool = False):
        """Integrate on (-inf,+inf) using Gauss-Hermite.

        If use_mpmath=True, uses full mpmath pipeline with integrate_mp and returns mp.mpf.
        Fallback paths also propagate use_mpmath to recursive _integrate_legendre calls.
        """
        from sympy import exp as sym_exp
        from hermite import GaussHermiteQuadrature

        v = Symbol(variable)

        if use_mpmath:
            quad = GaussHermiteQuadrature(n=n, use_mpmath=True, dps=80)
            weight = sym_exp(-v**2)
            stripped_expr = (expr / weight).simplify()

            result_mp = quad.integrate_mp(stripped_expr, dps=80)  # returns mp.mpf

            if abs(result_mp) > mp.mpf('1e30'):
                print(f"DEBUG _integrate_hermite: mpmath result too large — falling back to finite-window Legendre")
                L = self._compute_effective_support(expr, variable)
                osc_safe_n = self._compute_oscillation_safe_n(
                    expr, variable, -L, L, max(n * 2, 64)
                )
                return self._integrate_legendre(expr, -L, L, osc_safe_n, use_mpmath=True)

            return result_mp  # Return mp.mpf, NOT float(result_mp)
        else:
            stripped = lambdify(v, (expr / sym_exp(-v**2)).simplify(), modules="numpy")
            test_vals = stripped(np.array([3.0, 5.0, 7.0]))
            if np.any(np.abs(test_vals) > 1e6):
                func_original = lambdify(v, expr, modules="numpy")
                L = self._compute_effective_support(expr, variable)
                osc_safe_n = self._compute_oscillation_safe_n(
                    expr, variable, -L, L, max(n * 2, 64)
                )
                return self._integrate_legendre(func_original, -L, L, osc_safe_n)

            quad = GaussHermiteQuadrature(n=n, use_mpmath=False)
            return float(quad.integrate(stripped))

    def _compute_effective_support_laguerre(self, expr, variable: str,
                                             epsilon: float = 1e-15) -> float:
        v = Symbol(variable)
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_quadratic(inner, v):
                    coeff = abs(self._extract_x2_coefficient(inner.expand(), v))
                    return math.sqrt(math.log(1.0 / epsilon**2) / max(coeff, 1e-15))
            except Exception:
                pass
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if QuadratureAnalyzer._is_negative_linear(inner, v):
                    coeff = abs(self._extract_x_coefficient(inner, v))
                    return math.log(1.0 / epsilon) / max(coeff, 1e-15)
            except Exception:
                pass
        return 50.0

    # CATEGORY D FIX: _estimate_max_oscillation_frequency()
    def _estimate_max_oscillation_frequency(self, expr, variable: str,
                                              a: float, b: float) -> float:
        v = Symbol(variable)
        max_freq = 0.0

        trig_atoms = list(expr.atoms(sin)) + list(expr.atoms(cos))
        for sub in trig_atoms:
            try:
                phase = sub.args[0]
                dphase = diff(phase, v)
                for pt in (a, (a + b) / 2.0, b):
                    try:
                        val = abs(float(N(dphase.subs(v, pt), 15)))
                        if math.isfinite(val):
                            max_freq = max(max_freq, val)
                    except Exception:
                        pass
            except Exception:
                pass

        return max_freq

    # CATEGORY D FIX: _compute_oscillation_safe_n()
    def _compute_oscillation_safe_n(self, expr, variable: str,
                                      interval_a: float, interval_b: float,
                                      base_n: int) -> int:
        omega_max = self._estimate_max_oscillation_frequency(
            expr, variable, interval_a, interval_b
        )
        if omega_max < 1e-6:
            return base_n

        n_half_periods = omega_max * (interval_b - interval_a) / math.pi
        min_osc_nodes = int(10 * n_half_periods)

        return max(base_n, min(min_osc_nodes, 500))

    # CATEGORY C FIX: _extract_endpoint_singularity_exponents()
    def _extract_endpoint_singularity_exponents(self, expr, a: float, b: float) -> dict:
        v = self._var
        result = {"left": 0.0, "right": 0.0}
        for endpoint, side in [(a, "left"), (b, "right")]:
            try:
                distances = [1e-3, 1e-4, 1e-5, 1e-6]
                log_dists, log_vals = [], []
                for d in distances:
                    x_p = endpoint + d if side == "left" else endpoint - d
                    if not (a < x_p < b):
                        continue

                    fval = float(N(expr.subs(v, x_p), 15))
                    if math.isfinite(fval) and abs(fval) > 1e-20:
                        log_dists.append(math.log(d))
                        log_vals.append(math.log(abs(fval)))

                if len(log_dists) >= 3:
                    if log_vals[-1] > log_vals[0] + 0.1:
                        slope = np.polyfit(log_dists, log_vals, 1)[0]
                        alpha_est = -slope
                        if 0.01 < alpha_est < 0.99:
                            result[side] = alpha_est
            except Exception:
                continue
        return result

    # A6: Modified _integrate_jacobi with use_mpmath parameter
    def _integrate_jacobi(self, func_stripped, alpha: float, beta: float, n: int, a: float, b: float,
                           use_mpmath: bool = False) -> float:
        if use_mpmath:
            # Use mpmath.quad for Jacobi-weighted integration
            v = Symbol('t')
            try:
                f_mp = lambdify(v, func_stripped, modules='mpmath')
            except Exception:
                f_mp = func_stripped

            with mp.workdps(80):
                a_mp = mp.mpf(a)
                b_mp = mp.mpf(b)
                alpha_mp = mp.mpf(alpha)
                beta_mp = mp.mpf(beta)

                def integrand(x):
                    # Jacobi weight: (b-x)^alpha * (x-a)^beta on [a,b]
                    return ((b_mp - x)**alpha_mp) * ((x - a_mp)**beta_mp) * f_mp(x)

                try:
                    result = mp.quad(integrand, [a_mp, b_mp])
                    return float(result)
                except Exception:
                    # Fallback to scipy if mpmath quad fails
                    pass

        from scipy.special import roots_jacobi
        nodes_t, weights = roots_jacobi(n, -alpha, -beta)

        scale = (b - a) / 2.0
        shift = (b + a) / 2.0
        weight_scale = scale ** (1.0 - alpha - beta)

        total = 0.0
        for t, w in zip(nodes_t, weights):
            try:
                val = func_stripped(scale * float(t) + shift)
                if math.isfinite(val):
                    total += w * val
            except Exception:
                continue
        return float(total) * weight_scale

    # A6: Modified _integrate_with_jacobi_weight with use_mpmath parameter
    def _integrate_with_jacobi_weight(self, expr, variable, n, a, b, l_alpha, r_beta,
                                       use_mpmath: bool = False):
        from sympy import lambdify as sym_lambdify, powsimp

        v = Symbol(variable)
        g_expr = (expr * (v - a)**l_alpha * (b - v)**r_beta)
        g_expr_clean = powsimp(g_expr, combine='all', force=True)

        if use_mpmath:
            try:
                f_mp = sym_lambdify(v, g_expr_clean, modules='mpmath')
            except Exception:
                f_mp = sym_lambdify(v, g_expr_clean, modules="math")
            return self._integrate_jacobi(f_mp, r_beta, l_alpha, n, a, b, use_mpmath=True)

        try:
            func_stripped = sym_lambdify(v, g_expr_clean, modules=["numpy", "math"])
        except Exception:
            func_stripped = sym_lambdify(v, g_expr_clean, modules="math")

        return self._integrate_jacobi(func_stripped, r_beta, l_alpha, n, a, b, use_mpmath=False)

    # A5: Modified _integrate_laguerre with use_mpmath parameter
    def _integrate_laguerre(self, expr, variable: str, n: int, use_mpmath: bool = False):
        """Integrate on [0,+inf) using Gauss-Laguerre.

        If use_mpmath=True, uses full mpmath pipeline with integrate_mp and returns mp.mpf.
        Fallback paths also propagate use_mpmath to recursive _integrate_legendre calls.
        """
        from sympy import exp as sym_exp
        v = Symbol(variable)

        if use_mpmath:
            from laguerre import LaguerreQuadrature
            quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=True, dps=80)
            weight = sym_exp(-v)
            stripped_expr = (expr / weight).simplify()

            result_mp = quad.integrate_mp(stripped_expr, dps=80)  # returns mp.mpf

            if abs(result_mp) > mp.mpf('1e30'):
                print(f"DEBUG _integrate_laguerre: mpmath result too large — falling back to finite-window Legendre")
                L = self._compute_effective_support_laguerre(expr, variable)
                osc_safe_n = self._compute_oscillation_safe_n(
                    expr, variable, 0.0, L, max(n * 2, 64)
                )
                return self._integrate_legendre(expr, 0.0, L, osc_safe_n, use_mpmath=True)

            return result_mp  # Return mp.mpf, NOT float(result_mp)
        else:
            func_original = lambdify(v, expr, modules="numpy")

            weight = sym_exp(-v)
            stripped = lambdify(v, (expr / weight).simplify(), modules="numpy")

            test_x = np.array([10.0, 30.0, 60.0])
            raw_vals = stripped(test_x)
            test_vals = np.atleast_1d(np.asarray(raw_vals, dtype=np.float64))

            if np.any(~np.isfinite(test_vals)):
                print(f"DEBUG _integrate_laguerre: stripped integrand overflows at probe points — falling back to finite-window Legendre")
                L = self._compute_effective_support_laguerre(expr, variable)
                osc_safe_n = self._compute_oscillation_safe_n(
                    expr, variable, 0.0, L, max(n * 2, 64)
                )
                return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

            if len(test_vals) >= 2 and np.all(np.abs(test_vals) > 0):
                growth_ratio = abs(test_vals[-1]) / abs(test_vals[0])
                if growth_ratio > 1e6:
                    print(f"DEBUG _integrate_laguerre: stripped integrand grows (ratio={growth_ratio:.2e}) — falling back to finite-window Legendre")
                    L = self._compute_effective_support_laguerre(expr, variable)
                    osc_safe_n = self._compute_oscillation_safe_n(
                        expr, variable, 0.0, L, max(n * 2, 64)
                    )
                    return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

            omega_max = self._estimate_max_oscillation_frequency(expr, variable, 0.0, float("inf"))
            if omega_max > 1e-6:
                has_growth = False
                if len(test_vals) >= 2 and np.all(np.abs(test_vals) > 0):
                    growth_ratio = abs(test_vals[-1]) / abs(test_vals[0])
                    has_growth = growth_ratio > 1.5

                if has_growth:
                    print(f"DEBUG _integrate_laguerre: oscillations + stripped growth detected — falling back to finite-window Legendre")
                    L = self._compute_effective_support_laguerre(expr, variable)
                    osc_safe_n = self._compute_oscillation_safe_n(
                        expr, variable, 0.0, L, max(n * 2, 64)
                    )
                    return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

            from laguerre import LaguerreQuadrature
            quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=False)
            result = float(quad.integrate(stripped))

            if not math.isfinite(result):
                print(f"DEBUG _integrate_laguerre: result is non-finite — falling back to finite-window Legendre")
                L = self._compute_effective_support_laguerre(expr, variable)
                osc_safe_n = self._compute_oscillation_safe_n(
                    expr, variable, 0.0, L, max(n * 2, 64)
                )
                return self._integrate_legendre(func_original, 0.0, L, osc_safe_n)

            return result


# ---------------------------------------------------------------------------
#  Demo / self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

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