# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 09:14:44 2021

@author: pmarc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

donnee = pd.read_csv('DonneesFumeursv0r2.csv')
stats=donnee.describe()
dimensions=donnee.shape
nomsvariables = pd.DataFrame(donnee.columns)



"Théorème de Bayes"


Travailleurs_fumeurs=donnee[(donnee["Fumeur?"] == 'Oui')] 
Travailleurs_dans_quarantaine = donnee[((donnee["Âge"] <= 49) & (donnee["Âge"] >= 40))]
Fumeurs_dans_quarantaine=donnee[((donnee["Fumeur?"] == 'Oui') & (donnee["Âge"] <= 49) & (donnee["Âge"] >= 40))]

P_fumeurs = Travailleurs_fumeurs.shape[0]/donnee.shape[0]
P_quarantaine = Travailleurs_dans_quarantaine.shape[0]/donnee.shape[0]
Prob_quarantaine_étant_fumeur = Fumeurs_dans_quarantaine.shape[0]/Travailleurs_fumeurs.shape[0]


Prob_fumeur_étant_quarantaine = Prob_quarantaine_étant_fumeur*P_fumeurs/P_quarantaine

Prob_fumeur_étant_quarantaine_theo = Fumeurs_dans_quarantaine.shape[0]/Travailleurs_dans_quarantaine.shape[0]



