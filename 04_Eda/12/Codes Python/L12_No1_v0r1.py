# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 01:33:30 2021

@author: pmarc
Résolution Problème L12 - #1 par Pierre-Marc Juneau, 22 avril 2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
import statsmodels.api as sm
import scipy.stats as sts
import math

donnee = pd.read_csv('Immobilierv0r1.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)

X=donnee
MatriceR = X.corr()

X=X.drop('Prix par pied carré', 1)
Y=donnee["Prix par pied carré"]


"Estimé de la moyenne et de la variance pour Y"

Y_barre=Y.mean()
s=Y.std()
n=Y.shape[0]

mhu_min=Y_barre-sts.t.isf(0.05/2,n-1)*s/math.sqrt(n)
mhu_max=Y_barre+sts.t.isf(0.05/2,n-1)*s/math.sqrt(n)

Inter_mhu=sts.t.interval(0.95, n-1, loc=Y_barre, scale=s/math.sqrt(n))

sigma_est=s**2
sigma2_min=(n-1)*s**2/sts.chi2.isf(0.05/2,n-1)
sigma_min=math.sqrt(sigma2_min)

sigma2_max=(n-1)*s**2/sts.chi2.isf(1-0.05/2,n-1)
sigma_max=math.sqrt(sigma2_max)

ax=plt.hist(Y,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Y")


"Ronde 1"
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()


"Normalisation"

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler() 
Xnorm = scalerX.fit_transform(X.values)
scalerY = MinMaxScaler() 
Ynorm = scalerY.fit_transform(Y.values.reshape(-1,1))
Xnorm=pd.DataFrame(Xnorm)
Xnorm.columns=X.columns
Ynorm=pd.DataFrame(Ynorm)
Ynorm.columns=["Prix par pied carré"]

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
Ystand.columns=["Prix par pied carré"]

modele=sm.OLS(Ystand,Xstand.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(Xstand.assign(const=1))
resultats.summary()