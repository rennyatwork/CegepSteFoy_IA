# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 22:41:45 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
import statsmodels.api as sm
import scipy.stats as sts
# statsmodels.formula.api import ols
import math


donnee = pd.read_csv('DonnéesBiométriquesv0r2.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)


"Préparation des matrices"

X=donnee
X=pd.get_dummies(X)
X=X.drop({'Poids (kg)','Genre_Homme'},1)
Y=donnee['Poids (kg)']

MatriceR=(pd.get_dummies(donnee)).corr()

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"Intervalle de confiance autour de l'estimé de la grandeur moyenne et de la variance"

VarGrandeur=X['Grandeur (cm)']

ax=plt.hist(VarGrandeur,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Grandeur (cm)")

X_barre=VarGrandeur.mean()
s=VarGrandeur.std()
n=VarGrandeur.shape[0]

mhu_min=X_barre-sts.t.isf(0.05/2,n-1)*s/math.sqrt(n)
mhu_max=X_barre+sts.t.isf(0.05/2,n-1)*s/math.sqrt(n)

Inter_mhu=sts.t.interval(0.95, n-1, loc=X_barre, scale=s/math.sqrt(n))

sigma2_est=s**2
sigma2_min=(n-1)*s**2/sts.chi2.isf(0.05/2,n-1)
sigma_min=math.sqrt(sigma2_min)

sigma2_max=(n-1)*s**2/sts.chi2.isf(1-0.05/2,n-1)
sigma_max=math.sqrt(sigma2_max)


"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"Normalisation/Standardisation et régression multilinéaire"

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler() 
Xnorm = scalerX.fit_transform(X.values)
scalerY = MinMaxScaler() 
Ynorm = scalerY.fit_transform(Y.values.reshape(-1,1))
Xnorm=pd.DataFrame(Xnorm)
Xnorm.columns=X.columns
Ynorm=pd.DataFrame(Ynorm)
Ynorm.columns=["Poids (kg)"]

modele=sm.OLS(Ynorm,Xnorm.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(Xnorm.assign(const=1))
resultats.summary()


"Standardisation"

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
Xstand = scalerX.fit_transform(X.values)
scalerY = StandardScaler() 
Ystand = scalerY.fit_transform(Y.values.reshape(-1,1))
Xstand=pd.DataFrame(Xstand)
Xstand.columns=X.columns
Ystand=pd.DataFrame(Ystand)
Ystand.columns=["Poids (kg)"]

modele=sm.OLS(Ystand,Xstand.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(Xstand.assign(const=1))
resultats.summary()


