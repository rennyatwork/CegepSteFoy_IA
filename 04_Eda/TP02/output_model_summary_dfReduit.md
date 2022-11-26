key:  Production moyenne par entaille (L)
=========
var depend:  Production moyenne par entaille (L)
                                     OLS Regression Results                                    
===============================================================================================
Dep. Variable:     Production moyenne par entaille (L)   R-squared:                       0.910
Model:                                             OLS   Adj. R-squared:                  0.909
Method:                                  Least Squares   F-statistic:                 1.602e+04
Date:                                 Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                                         10:47:10   Log-Likelihood:                -4029.1
No. Observations:                                 1595   AIC:                             8062.
Df Residuals:                                     1593   BIC:                             8073.
Df Model:                                            1                                         
Covariance Type:                             nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
const                       -11.8088      0.316    -37.395      0.000     -12.428     -11.189
Nombre épisodes gel/dégel     1.8654      0.015    126.564      0.000       1.836       1.894
==============================================================================
Omnibus:                       34.465   Durbin-Watson:                   0.025
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               20.460
Skew:                          -0.110   Prob(JB):                     3.61e-05
Kurtosis:                       2.490   Cond. No.                         89.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
key:  Débit sève (L/j)
=========
var depend:  Débit sève (L/j)
                            OLS Regression Results                            
==============================================================================
Dep. Variable:       Débit sève (L/j)   R-squared:                       0.943
Model:                            OLS   Adj. R-squared:                  0.943
Method:                 Least Squares   F-statistic:                 2.623e+04
Date:                Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                        10:47:10   Log-Likelihood:                -13360.
No. Observations:                1595   AIC:                         2.672e+04
Df Residuals:                    1593   BIC:                         2.673e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
const                -3188.2215     46.670    -68.314      0.000   -3279.762   -3096.681
Temps bouilloire (h)  2628.9377     16.232    161.961      0.000    2597.099    2660.776
==============================================================================
Omnibus:                      278.374   Durbin-Watson:                   1.681
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1396.289
Skew:                           0.726   Prob(JB):                    6.30e-304
Kurtosis:                       7.347   Cond. No.                         5.53
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
key:  Sucre sève (%)
=========
var depend:  Sucre sève (%)
                            OLS Regression Results                            
==============================================================================
Dep. Variable:         Sucre sève (%)   R-squared:                       0.925
Model:                            OLS   Adj. R-squared:                  0.925
Method:                 Least Squares   F-statistic:                     9764.
Date:                Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                        10:47:10   Log-Likelihood:                 1192.9
No. Observations:                1595   AIC:                            -2380.
Df Residuals:                    1592   BIC:                            -2364.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
const                         1.4896      0.012    123.622      0.000       1.466       1.513
Précip. tot. (mm)            -0.0508      0.000   -104.089      0.000      -0.052      -0.050
Nombre épisodes gel/dégel     0.0517      0.001     92.563      0.000       0.051       0.053
==============================================================================
Omnibus:                       15.107   Durbin-Watson:                   1.148
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               10.823
Skew:                           0.079   Prob(JB):                      0.00447
Kurtosis:                       2.629   Cond. No.                         90.9
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
key:  Transmittance produit (%)
=========
var depend:  Transmittance produit (%)
                                OLS Regression Results                               
=====================================================================================
Dep. Variable:     Transmittance produit (%)   R-squared:                       0.683
Model:                                   OLS   Adj. R-squared:                  0.682
Method:                        Least Squares   F-statistic:                     683.8
Date:                       Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                               10:47:10   Log-Likelihood:                -5270.9
No. Observations:                       1595   AIC:                         1.055e+04
Df Residuals:                           1589   BIC:                         1.059e+04
Df Model:                                  5                                         
Covariance Type:                   nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
const                          -795.2753    520.302     -1.528      0.127   -1815.826     225.276
Osmoseur (heures opération/j)     1.1109      0.100     11.061      0.000       0.914       1.308
Température Bouilloire (0C)       9.6813      5.734      1.688      0.092      -1.566      20.928
Temps bouilloire (h)             -2.9453      0.257    -11.452      0.000      -3.450      -2.441
Sucre du sirop obtenu (%)        -2.1108      1.153     -1.831      0.067      -4.372       0.150
CategClasseSirop                 -9.2630      0.185    -49.946      0.000      -9.627      -8.899
==============================================================================
Omnibus:                      134.031   Durbin-Watson:                   1.981
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              278.568
Skew:                           0.535   Prob(JB):                     3.23e-61
Kurtosis:                       4.745   Cond. No.                     3.90e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.9e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
