# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:16:40 2021

@author: pmarc
Résolution Problème L10 - #2 par Pierre-Marc Juneau, 8 avril 2021

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
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)
donnee=donnee.dropna()
donnee = donnee.drop('Date', 1)
donnee2=pd.get_dummies(donnee)
MatriceCorr=donnee2.corr()


X=donnee
X=X.drop({'ID','Vins ($/2sem)'},1)
X=pd.get_dummies(X)
MatriceR = X.corr()


X = X.drop('Statut Mat_Autre', 1)
Y=donnee[["Vins ($/2sem)"]]
X=pd.get_dummies(X)
Xmat=X.assign(const=1).values
Ymat=Y.values.reshape(-1,1)
n=Xmat.shape[0]
p=Xmat.shape[1]

"Ronde 1"
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 2"
X=donnee
X=X.drop({'ID','Vins ($/2sem)','Statut Mat','Adolescents'},1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 3"
X=X.drop({'Temps depuis dernier achat','Joaillerie ($/2sem)'},1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 4"
X=X.drop({'Fruits ($/2sem)'},1)
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



