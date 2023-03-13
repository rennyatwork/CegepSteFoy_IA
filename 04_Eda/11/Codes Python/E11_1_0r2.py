# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:39:34 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as stm
import statsmodels.stats.weightstats as ws
import math

donnee = pd.read_csv('JoueursHockeyv0r1.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)

ax=plt.hist(donnee["+/-"],density=True, bins = 10, color = 'blue', edgecolor = 'black')
plt.xlabel("+/-")

ax = donnee.boxplot(by="Position", column="+/-")
ax.set_xlabel('')
ax.set_ylabel("+/-")


ValD=donnee[(donnee["Position"]=="D")]["+/-"].values.reshape(-1,1)
ValC=donnee[(donnee["Position"]=="C")]["+/-"].values.reshape(-1,1)
ValAG=donnee[(donnee["Position"]=="AG")]["+/-"].values.reshape(-1,1)
ValAD=donnee[(donnee["Position"]=="AD")]["+/-"].values.reshape(-1,1)

print(ValD.std())
print(ValC.std())
print(ValAG.std())
print(ValAD.std())


N=dimensions[0]
g=4
nD=ValD.shape[0]
nC=ValC.shape[0]
nAG=ValAG.shape[0]
nAD=ValAD.shape[0]
ytbar=donnee["+/-"].mean()
SSET=nD*((ValD.mean()-ytbar)**2)+nC*((ValC.mean()-ytbar)**2)+nAG*((ValAG.mean()-ytbar)**2)+nAD*((ValAD.mean()-ytbar)**2)
SSIT=((ValD-ValD.mean())**2).sum()+((ValC-ValC.mean())**2).sum()+((ValAG-ValAG.mean())**2).sum()+((ValAD-ValAD.mean())**2).sum()
SST=((ValD-ytbar)**2).sum()+((ValC-ytbar)**2).sum()+((ValAG-ytbar)**2).sum()+((ValAD-ytbar)**2).sum()
MSEET=SSET/(g-1)
MSEIT=SSIT/(N-g)
MSET=SST/(N-1)


F0=MSEET/MSEIT
CV=sts.f.isf(0.05,g-1,N-g)
p_value=sts.f.sf(F0,g-1,N-g)
p_value_fonction_directe=sts.f_oneway(ValD,ValC,ValAG,ValAD)


"Différence entre 2 moyennnes"

print(ValD.mean())
print(ValC.mean())
print(ValAG.mean())
print(ValAD.mean())

pvalue = sts.ttest_ind(ValD,ValC)

