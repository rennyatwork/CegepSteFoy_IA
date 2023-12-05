# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:30:52 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as stm


donnee = pd.read_csv('PersonnesActivesv0r2.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)



ax=donnee["Grandeur (cm)"].plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Grandeur (cm)")


sts.probplot(donnee["Grandeur (cm)"].values, dist=sts.norm, plot=plt.figure().add_subplot(111))


d=1000
grille_x = np.linspace(donnee["Grandeur (cm)"].min(), donnee["Grandeur (cm)"].max(), d)
dx=(donnee["Grandeur (cm)"].max()-(donnee["Grandeur (cm)"].min()))/(d-1)
mu, sigma = sts.norm.fit(donnee["Grandeur (cm)"].values)
pdf = sts.norm.pdf(grille_x, mu, sigma)
ax=donnee["Grandeur (cm)"].plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Grandeur (cm)")
ax.plot(grille_x, pdf, linewidth=3, color = 'red')
ax.set_ylabel("Fonction de densité f(x)")


Fit_normal = sts.kstest(donnee["Grandeur (cm)"],'norm',[mu, sigma])


A=np.random.normal(0,1,10000)
ax=pd.DataFrame(A).plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
Fit_normal2 = sts.kstest(A,'norm')


Z_scores=sts.zscore(donnee[["Âge","Poids (kg)","Grandeur (cm)"]].values,axis=0)

print(1-sts.norm(mu, sigma).cdf(180))

print(sts.norm(mu, sigma).pdf(171))


