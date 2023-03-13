# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:12:28 2021

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
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)

#X=donnee.drop('Espèces', 1)
X=pd.get_dummies(donnee)
X=X.drop('Espèces_Éperlan', 1)
MatriceR = X.corr()

X=X.drop('Poids (g)', 1)
Y=donnee["Poids (g)"]

"Développement des formules au long"

Xmat=X.assign(const=1).values
Ymat=Y.values.reshape(-1,1)
n=Xmat.shape[0]
p=Xmat.shape[1]

Beta_chap=np.dot(np.dot(np.linalg.inv(np.dot(Xmat.T,Xmat)),Xmat.T),Ymat)
Y_chap=np.dot(Xmat,Beta_chap)
C=np.linalg.inv(np.dot(Xmat.T,Xmat))
c_ii=np.diagonal(C)
y_bar=Ymat.mean()
plt.scatter(Ymat, Y_chap)
plt.xlabel("Y")
plt.ylabel("Y_chapeau")
plt.show()


SSR=((Y_chap-y_bar)**2).sum()
SSE=((Ymat-Y_chap)**2).sum()
SST=((Ymat-y_bar)**2).sum()
MSR=SSR/(p-1)
MSE=SSE/(n-p)
MST=SST/(n-1)
R2=SSR/SST
R2_ajust=1-(SSE/(n-p))/(SST/(n-1))
F0=MSR/MSE
p_value=sts.f.sf(abs(F0),p-1,n-p)


t0=Beta_chap/(MSE*c_ii.reshape(-1,1))**0.5
p_value=sts.t.sf(abs(t0),df=(n-p))*2
Betas_min=Beta_chap-sts.t.isf(0.05/2,n-p)*(MSE*c_ii.reshape(-1,1))**0.5
Betas_max=Beta_chap+sts.t.isf(0.05/2,n-p)*(MSE*c_ii.reshape(-1,1))**0.5
Compilation_Beta=pd.concat([pd.DataFrame(Betas_min),pd.DataFrame(Beta_chap),pd.DataFrame(Betas_max),pd.DataFrame(p_value)],axis=1)
Compilation_Beta.columns=['Min','Betas','Max','p-value']

for i in range(p-1):
    Compilation_Beta.rename(index={(i):X.columns[i]}, inplace=True)
Compilation_Beta.rename(index={(p-1):'Origine'}, inplace=True)


#Y_chap_min=[]
#Y_chap_max=[]
#Y_chap_p_value=[]
#for i in range(n):
#    Y_chap_min.append(Y_chap[i-1]-sts.t.isf(0.05/2,n-p)*(MSE*np.dot(np.dot(Xmat[i-1,:],C),Xmat[i-1,:].T))**0.5)
#    Y_chap_max.append(Y_chap[i-1]+sts.t.isf(0.05/2,n-p)*(MSE*np.dot(np.dot(Xmat[i-1,:],C),Xmat[i-1,:].T))**0.5)
#    t0_y=Y_chap[i-1]/(MSE*np.dot(np.dot(Xmat[i-1,:],C),Xmat[i-1,:].T))**0.5
#    Y_chap_p_value.append(sts.t.sf(abs(t0_y),df=(n-p))*2)


Epsilon=Ymat-Y_chap
plt.figure(1)
plt.plot(Ymat, Epsilon, 'o')
plt.xlabel("Y")
plt.ylabel("Résidus")
plt.show()
sts.probplot(Epsilon[:,0],dist=sts.norm, plot=plt.figure().add_subplot(111))
ax=plt.hist(Epsilon,density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("Erreurs")



"Avec les fonctions"

"Ronde 1"
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 2"
X=X.drop('Épaisseur (cm)', 1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
ychap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 3"
X=X.drop('Hauteur (cm)', 1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 4"
X=X.drop('Longueur 3 (cm)', 1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()

"Ronde 5"
X=X.drop('Longueur 1 (cm)', 1)
modele=sm.OLS(Y,X.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(X.assign(const=1))
resultats.summary()


import plotnine as p9

graph = p9.ggplot(data=donnee,
           mapping=p9.aes(x="Ymat", y="Y_chap", color='Espèces'))
print(graph + p9.geom_point())


sm.graphics.plot_ccpr(resultats,'Longueur 2 (cm)')

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
plt.xlabel("Y (Poids (g))")