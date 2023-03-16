# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:52:46 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sts
import math
from sklearn.decomposition import PCA

donnee = pd.read_csv('./Fichiers/Tomatesv0r1.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)

X=donnee
MatriceR=X.corr()

"Régression ronde 1"

X=X.drop('Indice de goût', 1)
Y=donnee["Indice de goût"]

modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Régression ronde 2"

Xsimple=X["Couleur (a-band)"]
Origine=pd.DataFrame(np.ones((dimensions[0],1), dtype=int))
Origine.columns=["Ordonnée à l'origine"]
Xsimple=pd.concat([Xsimple,Origine],axis=1)
Y=donnee["Indice de goût"]

modele=sm.OLS(Y,Xsimple)
resultats=modele.fit()
Y_chap = resultats.predict(Xsimple)
resultats.summary()



"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"VIF"

#X1=pd.DataFrame(X["Temps entreposage (jours)"].values.reshape(-1,1))
#X1.index=X.index
X1=X["Temps entreposage (jours)"]
X2=X["Couleur (a-band)"]
X3=X["Lycopene (mg/100g)"]

Origine=pd.DataFrame(np.ones((dimensions[0],1), dtype=int))
Origine.columns=["Ordonnée à l'origine"]
X1=pd.concat([X1,Origine],axis=1)
X2=pd.concat([X2,Origine],axis=1)
X3=pd.concat([X3,Origine],axis=1)

modele=sm.OLS(Y,X1)
resultats=modele.fit()
Y_chap = resultats.predict(X1)
resultats.summary()

modele=sm.OLS(Y,X2)
resultats=modele.fit()
Y_chap = resultats.predict(X2)
resultats.summary()

modele=sm.OLS(Y,X3)
resultats=modele.fit()
Y_chap = resultats.predict(X3)
resultats.summary()

"Approche avec VIF"

X1=X["Temps entreposage (jours)"]
X_not1=X[["Couleur (a-band)","Lycopene (mg/100g)"]]
modele=sm.OLS(X1,X_not1)
resultats=modele.fit()
resultats.summary()
VIF1=1/(1-resultats.rsquared)

X2=X["Couleur (a-band)"]
X_not2=X[["Temps entreposage (jours)","Lycopene (mg/100g)"]]
modele=sm.OLS(X2,X_not2)
resultats=modele.fit()
resultats.summary()
VIF2=1/(1-resultats.rsquared)

X3=X["Lycopene (mg/100g)"]
X_not3=X[["Couleur (a-band)","Temps entreposage (jours)"]]
modele=sm.OLS(X3,X_not3)
resultats=modele.fit()
resultats.summary()
VIF3=1/(1-resultats.rsquared)


from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
VIFPanda=pd.DataFrame(VIF)
VIFPanda.index=X.columns
VIFPanda.columns=["VIF"]

"Ronde 1"

X=X.drop('Couleur (a-band)', 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
VIFPanda=pd.DataFrame(VIF)
VIFPanda.index=X.columns
VIFPanda.columns=["VIF"]

modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 2"

X=X.drop('Temps entreposage (jours)', 1)

modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

