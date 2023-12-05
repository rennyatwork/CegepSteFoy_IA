# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:23:54 2021

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


"Test d'hypothèse sur la moyenne des poids des hommes"

PoidsHommes=donnee[(donnee["Genre"] == "Homme")]["Poids (kg)"]
stats_PoidsHommes=PoidsHommes.describe()
n_H=PoidsHommes.shape[0]
X_barre_H_Poids=PoidsHommes.mean()
s_H_Poids=PoidsHommes.std()


"Cas 1"

mhu0H=84
sigmaHPoids=math.sqrt(100)
Z0=(X_barre_H_Poids-mhu0H)/(sigmaHPoids/(math.sqrt(n_H)))
CV1 = sts.norm.isf(0.05/2)
p_value_calc1=sts.norm.sf(abs(Z0))*2
print(p_value_calc1)

#pvalue1=sts.ttest_1samp(PoidsHommes,84)

"Cas 2"
mhu0H=79
t0=(X_barre_H_Poids-mhu0H)/(s_H_Poids/(math.sqrt(n_H)))
CV2 = sts.t.isf(0.05/2,n_H-1)

p_value_calc2=sts.t.sf(abs(t0),df=(n_H-1))*2
pvalue2=sts.ttest_1samp(PoidsHommes,79)

"Comparaison grandeurs moyennes entre hommes et femmes"

"Cas 3"

GrandeursHommes=donnee[(donnee["Genre"] == "Homme")]["Grandeur (cm)"]
GrandeursFemmes=donnee[(donnee["Genre"] == "Femme")]["Grandeur (cm)"]
Stats_H=GrandeursHommes.describe()
Stats_F=GrandeursFemmes.describe()

X_barre_H_Grand=GrandeursHommes.mean()
X_barre_F_Grand=GrandeursFemmes.mean()
s_H_Grand=GrandeursHommes.std()
s_F_Grand=GrandeursFemmes.std()
n_H=GrandeursHommes.shape[0]
n_F=GrandeursFemmes.shape[0]


sigmaH=7.4
sigmaF=6.2
Z0=(X_barre_H_Grand-X_barre_F_Grand)/math.sqrt(sigmaH**2/n_H+sigmaF**2/n_F)
CV3 = sts.norm.isf(0.05/2)
p_value_calc3=sts.norm.sf(abs(Z0))*2
print(p_value_calc3)

#GrandeursHommesStats = stm.stats.DescrStatsW(GrandeursHommes)
#GrandeursFemmesStats = stm.stats.DescrStatsW(GrandeursFemmes)
    
#pval1= stm.stats.CompareMeans(GrandeursHommesStats,GrandeursFemmesStats).ztest_ind(usevar='unequal')


#SSE_H=sum(((GrandeursHommes.values-X_barre_H))**2)
#SSE_F=sum(((GrandeursFemmes.values-X_barre_F))**2)
#s_H=math.sqrt(SSE_H/(n_H-1))
#s_F=math.sqrt(SSE_F/(n_F-1))
#Sp=math.sqrt((SSE_H+SSE_F)/(n_H+n_F-2))


"Cas 4"

Sp=math.sqrt(((n_H-1)*s_H_Grand**2+(n_F-1)*s_F_Grand**2)/(n_H+n_F-2))
t0=(X_barre_H_Grand-X_barre_F_Grand)/(Sp*math.sqrt(1/n_H+1/n_F))
CV4 = sts.t.isf(0.05/2,n_H+n_F-2)
p_value_calc4=sts.t.sf(abs(t0),df=(n_H+n_F-2))*2
pvalue4 = sts.ttest_ind(GrandeursHommes,GrandeursFemmes)


"Cas 5"

v=(s_H_Grand**2/n_H+s_F_Grand**2/n_F)**2/((s_H_Grand**2/n_H)**2/(n_H+1)+(s_F_Grand**2/n_F)**2/(n_F+1))-2
t0=(X_barre_H_Grand-X_barre_F_Grand)/math.sqrt(s_H_Grand**2/n_H+s_F_Grand**2/n_F)
CV5 = sts.t.isf(0.05/2,v)
p_value_calc5=sts.t.sf(abs(t0),df=(v))*2




