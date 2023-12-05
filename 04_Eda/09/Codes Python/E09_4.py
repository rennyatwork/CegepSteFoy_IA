# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 09:14:44 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import random
import scipy.stats as sts
import matplotlib.pyplot as plt

donnee = pd.read_csv('DonneesFumeursv0r2.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)


"Génération des histogrammes et fonction de densité"


ax=donnee["Âge"].plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Âge")
ax.set_ylabel("Probabilités")


L=100000
k=100
Moyennes = []

Agevar=donnee["Âge"]
for i in range(L):
    Age_echantillon=random.choices(Agevar,weights=None, k=k)
    Moyennes.append(np.mean(Age_echantillon))

Moyennes_Ages=pd.DataFrame(Moyennes)

ax=Moyennes_Ages.plot.hist(density=True, bins = 100, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Âges moyens (xbarre)")
ax.set_ylabel("Probabilités")



sts.probplot(Moyennes_Ages[0].values, dist=sts.norm, plot=plt.figure().add_subplot(111))





