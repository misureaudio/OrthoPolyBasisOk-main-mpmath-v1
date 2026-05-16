+--------------------+-------------+-----------+----------------+---------------+--------------+--------+
| Test               | Family(D/A) | Conv(D/A) | Value (Direct) | Value (API)   | ErrEst (API) | Match? |
+--------------------+-------------+-----------+----------------+---------------+--------------+--------+
| Leg_Poly_Deg5      | Legen/Legen | Y/Y       | -4.996004e-16  | 1.197827e-15  | 6.08e-16     | ✅ YES |
| Leg_Poly_Deg10     | Legen/Legen | Y/Y       | 1.018182e+01   | 1.018182e+01  | 2.66e-14     | ✅ YES |
| Leg_Exp_Mild       | Legen/Legen | Y/Y       | 2.084381e+00   | 2.084381e+00  | 6.66e-15     | ✅ YES |
| Leg_Exp_Strong     | Legen/Legen | Y/Y       | 3.269083e+00   | 3.269083e+00  | 1.60e-14     | ✅ YES |
| Leg_Osc_Moderate   | Legen/Legen | Y/N       | 2.473396e-01   | 2.470869e-01  | 2.53e-04     | ❌ NO  |
| Leg_Osc_High       | Legen/Legen | Y/N       | -1.058814e-02  | -2.099246e-02 | 1.04e-02     | ❌ NO  |
| Leg_Sing_Alg       | Cheby/Cheby | N/N       | 1.814833e+00   | 1.814446e+00  | 3.26e-04     | ❌ NO  |
| Cheb_Tk_Exact      | Legen/Legen | Y/Y       | -1.398601e-02  | -1.398601e-02 | 9.78e-16     | ✅ YES |
| Cheb_Exp           | Legen/Legen | Y/Y       | 2.167382e+00   | 2.167382e+00  | 3.11e-15     | ✅ YES |
| Cheb_Osc_Weighted  | Cheby/Cheby | Y/N       | 9.129453e-02   | 9.129453e-02  | 1.40e-09     | ❌ NO  |
| Cheb_Sing_Beta     | Cheby/Cheby | N/N       | -3.441307e-01  | -3.443671e-01 | 1.94e-04     | ❌ NO  |
| Herm_Gauss_Exact   | Hermi/Hermi | Y/Y       | 1.772454e+00   | 1.772454e+00  | 1.33e-15     | ✅ YES |
| Herm_Perturbed     | Hermi/Hermi | Y/Y       | 1.944465e+02   | 1.944465e+02  | 0.00e+00     | ✅ YES |
| Herm_Osc_Gauss     | Legen/Legen | N/N       | nan            | None          | None         | ✅ YES |
| Herm_Poly_Gauss    | Hermi/Hermi | Y/Y       | 3.323351e+00   | 3.323351e+00  | 1.64e-14     | ✅ YES |
| Lag_Exp_Exact      | Lague/Lague | Y/Y       | 1.666667e+00   | 1.666667e+00  | 0.00e+00     | ✅ YES |
| Lag_Poly_Exp       | Lague/Lague | Y/N       | 4.018776e+01   | 4.018776e+01  | 8.13e-08     | ❌ NO  |
| Lag_Osc_Decay      | Lague/Lague | Y/Y       | 4.844626e-03   | 4.844626e-03  | 0.00e+00     | ✅ YES |
| Lag_Log_Sing       | Lague/Lague | Y/Y       | 1.845821e+00   | 1.845821e+00  | 0.00e+00     | ✅ YES |
| Cross_1over1pX4    | Hermi/Hermi | N/Y       | 2.273919e+00   | 2.273919e+00  | 0.00e+00     | ❌ NO  |
| Stress_RapidOsc    | Legen/Legen | Y/N       | 1.548492e-03   | -4.487882e-02 | 6.69e-02     | ❌ NO  |
| Stress_NarrowPeak  | Legen/Legen | N/N       | 4.439938e-01   | 4.441234e-01  | 1.25e-04     | ❌ NO  |
| Stress_HighFreqLag | Lague/Lague | Y/Y       | 1.999352e-02   | 1.999352e-02  | 0.00e+00     | ✅ YES |
| Leg_HighPoly       | Legen/Legen | Y/Y       | 8.000000e-01   | 8.000000e-01  | 8.88e-16     | ✅ YES |
| Leg_Osc_ω5         | Legen/Legen | Y/Y       | -3.835697e-01  | -3.835697e-01 | 8.88e-16     | ✅ YES |
| Leg_Osc_ω25        | Legen/Legen | Y/N       | -1.058814e-02  | 7.379048e-01  | 7.59e-01     | ❌ NO  |
| Leg_Osc_ω80        | Legen/Legen | N/N       | -3.942381e-01  | -2.923117e-01 | 5.23e-01     | ❌ NO  |
| Lag_Exp_c0.3       | Lague/Lague | Y/Y       | 3.333333e+00   | 3.333333e+00  | 1.33e-15     | ✅ YES |
| Lag_Exp_c1.1       | Lague/Lague | Y/Y       | 9.090909e-01   | 9.090909e-01  | 5.55e-16     | ✅ YES |
| Lag_Exp_c2.5       | Lague/Lague | Y/N       | 4.000000e-01   | 3.999966e-01  | 3.37e-06     | ❌ NO  |
| Leg_ExplicitN16    | Legen/Legen | Y/Y       | 6.666667e-01   | 6.666667e-01  | 2.11e-15     | ✅ YES |
| Lag_ExplicitN32    | Lague/Lague | Y/Y       | 1.000000e+00   | 1.000000e+00  | 6.66e-16     | ✅ YES |
| Herm_HighPrec      | Hermi/Hermi | Y/Y       | 1.772454e+00   | 1.772454e+00  | 4.44e-16     | ✅ YES |
+--------------------+-------------+-----------+----------------+---------------+--------------+--------+

Fidelity Score: 21/33 tests passed 1-to-1 mapping.