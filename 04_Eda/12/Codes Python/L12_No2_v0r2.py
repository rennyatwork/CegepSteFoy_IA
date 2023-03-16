# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:16:40 2021

@author: pmarc
Résolution Problème L12 - #2 par Pierre-Marc Juneau, 22 avril 2021

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
import statsmodels.api as sm
import scipy.stats as sts
# statsmodels.formula.api import ols


donnee = pd.read_csv('DonneesMarketingv0r3.csv')

"Pré-traitement"

donnee=donnee.dropna()

Q1 = donnee["Revenus"].quantile(0.25)
Q3 = donnee["Revenus"].quantile(0.75)
IQR = Q3 - Q1
donnee = donnee[(donnee["Revenus"] > (Q1 - 1.5 * IQR)) & (donnee["Revenus"] < (Q3 + 1.5 * IQR))]


stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)
donnee=donnee.dropna()
donnee = donnee.drop('Date', 1)
MatriceCorr=donnee.corr()


X=donnee
X=X.drop({'ID','Vins ($/2sem)'},1)
X=pd.get_dummies(X)
X = X.drop('Statut Mat_Autre', 1)
Y=donnee[["Vins ($/2sem)"]]
X=pd.get_dummies(X)
Xmat=X.assign(const=1).values
Ymat=Y.values.reshape(-1,1)
n=Xmat.shape[0]
p=Xmat.shape[1]

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"Régression multilinéaire"

"Ronde 1"
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 2"
X=donnee
X=X.drop({'ID','Vins ($/2sem)','Âge','Temps depuis dernier achat','Statut Mat'},1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 3"
X=X.drop({'Achats avec rabais','Joaillerie ($/2sem)'},1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()


plt.scatter(Ymat, Y_chap)
plt.xlabel("Y")
plt.ylabel("Y_chapeau")
plt.show()

Epsilon=Ymat-Y_chap.values.reshape(-1,1)
plt.figure(1)
plt.plot(Ymat, Epsilon, 'o')
plt.xlabel("Y")
plt.ylabel("Résidus")
plt.show()
sts.probplot(Epsilon[:,0],dist=sts.norm, plot=plt.figure().add_subplot(111))
ax=plt.hist(Epsilon,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Erreurs")

ax=plt.hist(Ymat,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Y")

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"En utilisant une transformation"

from sklearn.preprocessing import PowerTransformer
ptX = PowerTransformer()
ptX.fit(X)
Xtransf=pd.DataFrame(ptX.transform(X))
Xtransf.columns=X.columns
ptY = PowerTransformer()
ptY.fit(Y)
Ytransf=pd.DataFrame(ptY.transform(Y))
Ytransf.columns=Ytransf.columns
modele=sm.OLS(Ytransf,Xtransf)
resultats=modele.fit()
Y_chap_transf = resultats.predict(Xtransf)
resultats.summary()

ax=plt.hist(Ytransf,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Y")

plt.scatter(Ytransf, Y_chap_transf)
plt.xlabel("Y")
plt.ylabel("Y_chapeau")
plt.show()

Y_chapeau = ptY.inverse_transform(Y_chap_transf.values.reshape(-1,1))
plt.scatter(Y, Y_chapeau)
plt.xlabel("Y - Vins ($/2sem)")
plt.ylabel("Y_chapeau - Vins ($/2sem)")
plt.show()
R2=np.corrcoef(Y.T,Y_chap.T)[0,1]

Epsilon=Y.values.reshape(-1,1)-Y_chapeau.reshape(-1,1)
plt.figure(1)
plt.plot(Y.values, Epsilon, 'o')
plt.xlabel("Y")
plt.ylabel("Résidus")
plt.show()