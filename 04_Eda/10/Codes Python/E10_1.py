# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 09:52:08 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as stm
import statsmodels.stats.weightstats as ws
import math


donnee = pd.read_csv('PersonnesActivesv0r2.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)
Variable=donnee["Âge"]
ax=Variable.plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Âge")
ax.set_ylabel("Fonction de densité f(x)")

"Vérifier si les données suivent une loi normale"


d=1000
grille_x = np.linspace(Variable.min(), Variable.max(), d)
dx=(Variable.max()-(Variable.min()))/(d-1)
mu, sigma = sts.norm.fit(Variable.values)
param=sts.norm.fit(Variable.values)
pdf = sts.norm.pdf(grille_x, mu, sigma)
ax=Variable.plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Âge")
ax.plot(grille_x, pdf, linewidth=3, color = 'red')
ax.set_ylabel("Fonction de densité f(x)")


sts.probplot(Variable.values, dist=sts.norm(mu, sigma), plot=plt.figure().add_subplot(111))
Fit_normal = sts.kstest(Variable,'norm',param)



"Vérifier si les données suivent une loi gamma"

d=1000
grille_x = np.linspace(Variable.min(), Variable.max(), d)
dx=(Variable.max()-(Variable.min()))/(d-1)
shape,loc,scale=sts.gamma.fit(Variable.values,loc=0.1)
param=sts.gamma.fit(Variable.values)
pdf = sts.gamma.pdf(grille_x,shape,loc,scale)
ax=Variable.plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Âge")
ax.plot(grille_x, pdf, linewidth=3, color = 'red')
ax.set_ylabel("Fonction de densité f(x)")


sts.probplot(Variable.values, dist=sts.gamma(shape,loc,scale), plot=plt.figure().add_subplot(111))
Fit_gamma = sts.kstest(Variable.values,'gamma',param)