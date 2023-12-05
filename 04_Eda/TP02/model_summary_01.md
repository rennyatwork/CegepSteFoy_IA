---- Sommaire dfNumerique ----
=========
var depend:  Production moyenne par entaille (L)
                                     OLS Regression Results                                    
===============================================================================================
Dep. Variable:     Production moyenne par entaille (L)   R-squared:                       0.948
Model:                                             OLS   Adj. R-squared:                  0.947
Method:                                  Least Squares   F-statistic:                     1675.
Date:                                 Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                                         10:37:26   Log-Likelihood:                -3594.8
No. Observations:                                 1595   AIC:                             7226.
Df Residuals:                                     1577   BIC:                             7322.
Df Model:                                           17                                         
Covariance Type:                             nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
const                            -0.2839      0.365     -0.779      0.436      -0.999       0.431
Année                             0.1457      0.016      8.976      0.000       0.114       0.178
Jour Calendrier Saison           -0.0043      0.004     -1.133      0.257      -0.012       0.003
Temp max.(°C)                    -0.8091      0.310     -2.608      0.009      -1.417      -0.201
Temp min.(°C)                    -0.0123      0.348     -0.035      0.972      -0.695       0.670
Temp moy.(°C)                     0.8346      0.463      1.803      0.072      -0.074       1.743
Diff Temp (°C)                    0.3788      0.234      1.619      0.106      -0.080       0.838
Précip. tot. (mm)                -0.0130      0.011     -1.180      0.238      -0.035       0.009
Précip. Tot. Hiver (mm)           0.0094      0.001     12.093      0.000       0.008       0.011
Nombre épisodes gel/dégel         1.8745      0.014    138.095      0.000       1.848       1.901
Alimentation osmoseur (L/j)       0.0047      0.045      0.104      0.918      -0.084       0.094
Osmoseur (heures opération/j)    -1.9619     20.120     -0.098      0.922     -41.427      37.503
Pression osmoseur (bar)          -4.5324      5.832     -0.777      0.437     -15.973       6.908
Sucre sortie osmoseur (%)         4.5522      5.832      0.781      0.435      -6.887      15.991
Température Bouilloire (0C)      -1.8524      2.019     -0.917      0.359      -5.813       2.108
Temps bouilloire (h)              0.5337      0.172      3.099      0.002       0.196       0.872
Sucre du sirop obtenu (%)         0.4001      0.406      0.986      0.324      -0.396       1.196
Quantité de sirop obtenue (L)    -0.0091      0.003     -2.901      0.004      -0.015      -0.003
CategClasseSirop                 -0.0084      0.069     -0.121      0.904      -0.144       0.127
==============================================================================
Omnibus:                       72.149   Durbin-Watson:                   0.065
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               36.509
Skew:                          -0.181   Prob(JB):                     1.18e-08
Kurtosis:                       2.353   Cond. No.                     2.45e+18
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.92e-27. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
=========
var depend:  Débit sève (L/j)
                            OLS Regression Results                            
==============================================================================
Dep. Variable:       Débit sève (L/j)   R-squared:                       0.988
Model:                            OLS   Adj. R-squared:                  0.988
Method:                 Least Squares   F-statistic:                     7498.
Date:                Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                        10:37:26   Log-Likelihood:                -12128.
No. Observations:                1595   AIC:                         2.429e+04
Df Residuals:                    1577   BIC:                         2.439e+04
Df Model:                          17                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
const                            88.1231     76.789      1.148      0.251     -62.497     238.743
Année                            -2.1643      3.420     -0.633      0.527      -8.873       4.544
Jour Calendrier Saison           -2.6217      0.793     -3.307      0.001      -4.177      -1.066
Temp max.(°C)                   -44.2617     65.346     -0.677      0.498    -172.437      83.913
Temp min.(°C)                   -40.0538     73.321     -0.546      0.585    -183.871     103.764
Temp moy.(°C)                    93.1962     97.546      0.955      0.340     -98.137     284.530
Diff Temp (°C)                   23.8307     49.284      0.484      0.629     -72.839     120.501
Précip. tot. (mm)                13.9818      2.327      6.009      0.000       9.418      18.546
Précip. Tot. Hiver (mm)           0.3227      0.164      1.972      0.049       0.002       0.644
Nombre épisodes gel/dégel       -24.5462      2.860     -8.584      0.000     -30.155     -18.937
Alimentation osmoseur (L/j)       1.5156      9.540      0.159      0.874     -17.197      20.228
Osmoseur (heures opération/j)  -347.3460   4238.557     -0.082      0.935   -8661.147    7966.455
Pression osmoseur (bar)        1420.3643   1228.687      1.156      0.248    -989.668    3830.397
Sucre sortie osmoseur (%)     -1399.5745   1228.579     -1.139      0.255   -3809.394    1010.245
Température Bouilloire (0C)    -481.5438    425.368     -1.132      0.258   -1315.891     352.803
Temps bouilloire (h)            740.9638     36.284     20.421      0.000     669.793     812.135
Sucre du sirop obtenu (%)       116.6203     85.462      1.365      0.173     -51.011     284.251
Quantité de sirop obtenue (L)    14.1561      0.663     21.336      0.000      12.855      15.458
CategClasseSirop                118.4224     14.591      8.116      0.000      89.803     147.042
==============================================================================
Omnibus:                      208.571   Durbin-Watson:                   1.587
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1873.780
Skew:                           0.264   Prob(JB):                         0.00
Kurtosis:                       8.283   Cond. No.                     2.45e+18
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.92e-27. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
=========
var depend:  Sucre sève (%)
                            OLS Regression Results                            
==============================================================================
Dep. Variable:         Sucre sève (%)   R-squared:                       0.959
Model:                            OLS   Adj. R-squared:                  0.959
Method:                 Least Squares   F-statistic:                     2195.
Date:                Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                        10:37:26   Log-Likelihood:                 1687.3
No. Observations:                1595   AIC:                            -3339.
Df Residuals:                    1577   BIC:                            -3242.
Df Model:                          17                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
const                            -0.0062      0.013     -0.465      0.642      -0.032       0.020
Année                         -9.812e-05      0.001     -0.166      0.868      -0.001       0.001
Jour Calendrier Saison           -0.0001      0.000     -0.995      0.320      -0.000       0.000
Temp max.(°C)                    -0.0281      0.011     -2.485      0.013      -0.050      -0.006
Temp min.(°C)                    -0.0261      0.013     -2.057      0.040      -0.051      -0.001
Temp moy.(°C)                     0.0542      0.017      3.211      0.001       0.021       0.087
Diff Temp (°C)                    0.0004      0.009      0.052      0.958      -0.016       0.017
Précip. tot. (mm)                -0.0493      0.000   -122.309      0.000      -0.050      -0.048
Précip. Tot. Hiver (mm)          -0.0006   2.83e-05    -21.155      0.000      -0.001      -0.001
Nombre épisodes gel/dégel         0.0491      0.000     99.202      0.000       0.048       0.050
Alimentation osmoseur (L/j)       0.0022      0.002      1.338      0.181      -0.001       0.005
Osmoseur (heures opération/j)    -0.9922      0.734     -1.353      0.176      -2.431       0.447
Pression osmoseur (bar)          -0.0990      0.213     -0.466      0.641      -0.516       0.318
Sucre sortie osmoseur (%)         0.0988      0.213      0.465      0.642      -0.318       0.516
Température Bouilloire (0C)       0.0613      0.074      0.832      0.405      -0.083       0.206
Temps bouilloire (h)             -0.0181      0.006     -2.885      0.004      -0.030      -0.006
Sucre du sirop obtenu (%)        -0.0156      0.015     -1.054      0.292      -0.045       0.013
Quantité de sirop obtenue (L)     0.0006      0.000      4.923      0.000       0.000       0.001
CategClasseSirop                 -0.0040      0.003     -1.567      0.117      -0.009       0.001
==============================================================================
Omnibus:                      594.011   Durbin-Watson:                   2.071
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               80.205
Skew:                           0.049   Prob(JB):                     3.83e-18
Kurtosis:                       1.906   Cond. No.                     2.45e+18
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.92e-27. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
=========
var depend:  Transmittance produit (%)
                                OLS Regression Results                               
=====================================================================================
Dep. Variable:     Transmittance produit (%)   R-squared:                       0.710
Model:                                   OLS   Adj. R-squared:                  0.707
Method:                        Least Squares   F-statistic:                     226.8
Date:                       Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                               10:37:26   Log-Likelihood:                -5200.0
No. Observations:                       1595   AIC:                         1.044e+04
Df Residuals:                           1577   BIC:                         1.053e+04
Df Model:                                 17                                         
Covariance Type:                   nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
const                            -1.4851      0.997     -1.489      0.137      -3.441       0.471
Année                             0.0938      0.044      2.112      0.035       0.007       0.181
Jour Calendrier Saison           -0.0626      0.010     -6.080      0.000      -0.083      -0.042
Temp max.(°C)                    -0.8227      0.849     -0.969      0.332      -2.487       0.842
Temp min.(°C)                    -0.9644      0.952     -1.013      0.311      -2.832       0.903
Temp moy.(°C)                     1.7313      1.267      1.367      0.172      -0.753       4.216
Diff Temp (°C)                   -0.1872      0.640     -0.293      0.770      -1.443       1.068
Précip. tot. (mm)                -0.0419      0.030     -1.388      0.165      -0.101       0.017
Précip. Tot. Hiver (mm)          -0.0043      0.002     -2.007      0.045      -0.008   -9.73e-05
Nombre épisodes gel/dégel         0.0856      0.037      2.306      0.021       0.013       0.158
Alimentation osmoseur (L/j)      -0.2928      0.124     -2.363      0.018      -0.536      -0.050
Osmoseur (heures opération/j)   132.0475     55.044      2.399      0.017      24.081     240.014
Pression osmoseur (bar)         -23.7710     15.956     -1.490      0.136     -55.069       7.527
Sucre sortie osmoseur (%)        23.7526     15.955      1.489      0.137      -7.542      55.048
Température Bouilloire (0C)       7.3236      5.524      1.326      0.185      -3.512      18.159
Temps bouilloire (h)             -0.9182      0.471     -1.949      0.052      -1.842       0.006
Sucre du sirop obtenu (%)        -1.7523      1.110     -1.579      0.115      -3.929       0.425
Quantité de sirop obtenue (L)    -0.0458      0.009     -5.317      0.000      -0.063      -0.029
CategClasseSirop                 -8.5283      0.189    -45.008      0.000      -8.900      -8.157
==============================================================================
Omnibus:                      117.307   Durbin-Watson:                   2.087
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              270.344
Skew:                           0.443   Prob(JB):                     1.98e-59
Kurtosis:                       4.812   Cond. No.                     2.45e+18
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.92e-27. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
