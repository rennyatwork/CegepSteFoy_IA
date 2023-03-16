# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:12:54 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
import statsmodels.api as sm
import scipy.stats as sts
import math

donnee = pd.read_csv('Poissonsv0r2.csv')
donnee=donnee.dropna()
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)

X=donnee.drop({'Espèces','Poids (g)'}, 1)
Y=donnee['Poids (g)']
Yclass=donnee['Espèces']

"Ronde 1"
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 2"
X2=X.drop({'Longueur 2 (cm)','Épaisseur (cm)'}, 1)
modele=sm.OLS(Y,X2.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X2.assign(const=1))
resultats.summary()

plt.scatter(Y, Y_chap)
plt.xlabel("Y")
plt.ylabel("Y_chapeau")
plt.show()

Epsilon=Y.values.reshape(-1,1)-Y_chap.values.reshape(-1,1)
plt.figure(1)
plt.plot(Y.values, Epsilon, 'o')
plt.xlabel("Y")
plt.ylabel("Résidus")
plt.show()



"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"Transformation Box-Cox"

ax=plt.hist(Y,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Poids (g)")

X_BC=X2[(Y != 0)]
Y_BC=Y[(Y != 0)]

from sklearn.preprocessing import PowerTransformer
ptY_BC = PowerTransformer(method='box-cox')
ptY_BC.fit(Y_BC.values.reshape(-1,1))
lambdasY=ptY_BC.lambdas_
Ytransf_BC=ptY_BC.transform(Y_BC.values.reshape(-1,1))

ax=plt.hist(Ytransf_BC,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Poids (g) Transformé Box-Cox")

ptX_BC = PowerTransformer(method='box-cox')
ptX_BC.fit(X_BC.values)
lambdasX=ptX_BC.lambdas_
Xtransf_BC=ptX_BC.transform(X_BC.values)


modele=sm.OLS(Ytransf_BC,X_BC.assign(const=1))
resultats=modele.fit()
Ytransf_chap = resultats.predict(X_BC.assign(const=1))
resultats.summary()
plt.scatter(Ytransf_BC, Ytransf_chap)
plt.xlabel("Ytransf")
plt.ylabel("Ytransf_chapeau")
plt.show()

Y_BC_chapeau = ptY_BC.inverse_transform(Ytransf_chap.values.reshape(-1,1))
plt.scatter(Y_BC, Y_BC_chapeau)
plt.xlabel("Y_BC - Poids (g)")
plt.ylabel("Y_BC_chapeau - Poids (g)")
plt.show()

Epsilon=Y_BC.values.reshape(-1,1)-Y_BC_chapeau.reshape(-1,1)
plt.figure(1)
plt.plot(Y_BC.values, Epsilon, 'o')
plt.xlabel("Y")
plt.ylabel("Résidus")
plt.show()

sts.probplot(Epsilon[:,0],dist=sts.norm, plot=plt.figure().add_subplot(111))
