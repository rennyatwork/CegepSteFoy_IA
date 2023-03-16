# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:49:26 2021

@author: pmarc
Résolution Problème L11 - #1 par Pierre-Marc Juneau, 17 avril 2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
import statsmodels.api as sm
import scipy.stats as sts
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor

donnee = pd.read_csv('Poissonsv0r2.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)

#X=donnee.drop('Espèces', 1)
X=pd.get_dummies(donnee)
X=X.drop('Espèces_Éperlan', 1)
MatriceR = X.corr()

X=X.drop('Poids (g)', 1)
Y=donnee["Poids (g)"]

"Ronde 1"

VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
VIFPanda=pd.DataFrame(VIF)
VIFPanda.index=X.columns
VIFPanda.columns=["VIF"]

modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"VIFs très élevés pour les 3 longueurs (même ordre de grandeur), et il y a des coefficients non-significatifs"
"Garder une seule longueur (elles sont très corrélées ensembles, donc capturent la même tendance)"
"Garder la longueur avec la plus petite p-value (par exemple)"

"Ronde 2"

X=X.drop({'Longueur 2 (cm)','Longueur 3 (cm)'}, 1)
VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
VIFPanda=pd.DataFrame(VIF)
VIFPanda.index=X.columns
VIFPanda.columns=["VIF"]

modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"VIFs élevés pour hauteur et épaisseur, avec des coefficients non-significatifs"
"Enlever ces variables"

"Ronde 3"

X=X.drop({'Hauteur (cm)','Épaisseur (cm)'}, 1)
VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
VIFPanda=pd.DataFrame(VIF)
VIFPanda.index=X.columns
VIFPanda.columns=["VIF"]

modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Tous les Betas sont significatifs, modèle et R2 intéressants"