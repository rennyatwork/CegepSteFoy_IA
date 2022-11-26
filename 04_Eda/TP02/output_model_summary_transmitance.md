------
dependent var y:  Transmittance produit (%)
n:  0
score:  0.6894251136364131
high_score:  0
rfe.support_:  [False False False False False  True]
rfe.ranking_:  [4 2 3 6 5 1]
n:  1
score:  0.7042969991851935
high_score:  0.6894251136364131
rfe.support_:  [False  True False False False  True]
rfe.ranking_:  [3 1 2 5 4 1]
Optimum number of features: 2
Score with 2 features: 0.704297
df columns to keep: 
shape
(1595, 2)
                  colName  keep
1  Jour Calendrier Saison  True
5        CategClasseSirop  True
=========
var depend:  Transmittance produit (%)
                                OLS Regression Results                               
=====================================================================================
Dep. Variable:     Transmittance produit (%)   R-squared:                       0.668
Model:                                   OLS   Adj. R-squared:                  0.668
Method:                        Least Squares   F-statistic:                     1602.
Date:                       Sat, 26 Nov 2022   Prob (F-statistic):               0.00
Time:                               10:56:13   Log-Likelihood:                -5307.0
No. Observations:                       1595   AIC:                         1.062e+04
Df Residuals:                           1592   BIC:                         1.064e+04
Df Model:                                  2                                         
Covariance Type:                   nonrobust                                         
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
const                     70.3492      0.532    132.189      0.000      69.305      71.393
Jour Calendrier Saison    -0.0571      0.007     -7.765      0.000      -0.072      -0.043
CategClasseSirop          -9.5795      0.186    -51.562      0.000      -9.944      -9.215
==============================================================================
Omnibus:                       83.425   Durbin-Watson:                   2.093
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              134.240
Skew:                           0.425   Prob(JB):                     7.08e-30
Kurtosis:                       4.139   Cond. No.                         181.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
