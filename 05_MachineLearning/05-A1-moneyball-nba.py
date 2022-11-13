#!/usr/bin/env python
# coding: utf-8

# **420-A52-SF - Algorithmes d'apprentissage supervisé - Automne 2022 - Spécialisation technique en Intelligence Artificielle**<br/>
# MIT License - Copyright (c) 2022 Mikaël Swawola
# <br/>
# ![Travaux Pratiques - Moneyball NBA](static/05-A1-banner.png)
# <br/>
# **Objectif:** cette séance de travaux pratique est consacrée à la mise en oeuvre de l'ensemble des connaissances acquises jusqu'alors sur un nouveau jeu de données, *NBA*

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 0 - Chargement des bibliothèques

# In[2]:


# Manipulation de données
import numpy as np
import pandas as pd

# Visualisation de données
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Configuration de la visualisation
sns.set(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})


# ### 1 - Lecture du jeu de données *NBA*

# **Lire le fichier `NBA_train.csv`**

# In[5]:


# Compléter le code ci-dessous ~ 1 ligne
NBA = pd.read_csv("../../data/NBA_train.csv")


# **Afficher les dix premières lignes de la trame de données**

# In[6]:


# Compléter le code ci-dessous ~ 1 ligne
NBA.head()


# Ci-dessous, la description des différentes variables explicatives du jeu de données
# </br>
# 
# | Variable      | Description                                                   |
# | ------------- |:-------------------------------------------------------------:|
# | SeasonEnd     | Année de fin de la saison                                     |
# | Team          | Nom de l'équipe                                               |
# | Playoffs      | Indique si l'équipe est allée en playoffs                     |
# | W             | Nombre de victoires au cours de la saison régulière           |
# | PTS           | Nombre de points obtenus (saison régulière)                   |
# | oppPTS        | Nombre de points obtenus pas les opposants (saison régulière) |
# | FG            | Nombre de Field Goals réussis                                 |
# | FGA           | Nombre de tentatives de Field Goals                           |
# | 2P            | Nombre de 2-pointers réussis                                  |
# | 2PA           | Nombre de tentatives de 2-pointers                            |
# | 3P            | Nombre de 3-pointers réussis                                  |
# | 3PA           | Nombre de tentatives de 3-pointers                            |
# | FT            | Nombre de Free throws réussis                                 |
# | FTA           | Nombre de tentatives de Free throws                           |
# | ORB           | Nombre de rebonds offensifs                                   |
# | DRB           | Nombre de rebonds défensifs                                   |
# | AST           | Nombre de passes décisives (assists)                          |
# | STL           | Nombre d'interceptions (steals)                               |
# | BLK           | Nombre de contres (blocks)                                    |
# | TOV           | Nombre de turnovers                                           |
# 

# ### 1 - Régression linéaire simple

# Nous allons dans un premier temps effectuer la prédiction du nombre de victoires au cours de la saison régulière en fonction de la différence de points obtenus pas l'équipe et par ses opposants
# <br/><br/>
# Nous commencons donc par un peu d'**ingénierie de données**. Une nouvelle variable explicative correspondant à la différence de points obtenus pas l'équipe et par ses opposants est crée

# **Créer un nouvelle variable PTSdiff, représentant la différence entre PTS et oppPTS**

# In[7]:


# Compléter le code ci-dessous ~ 1 ligne
NBA['PTSdiff'] = NBA['PTS'] - NBA['oppPTS']


# In[8]:


NBA.head()


# **Stocker le nombre de lignes du jeu de donnée (nombre d'exemples d'entraînement) dans la variable `m`**

# In[9]:


# Compléter le code ci-dessous ~ 1 ligne
m = len(NBA)


# **Stocker le nombre de victoires au cours de la saison dans la variable `y`. Il s'agira de la variable que l'on cherche à prédire**

# In[10]:


m


# In[11]:


# Compléter le code ci-dessous ~ 1 ligne
y = NBA['W'].values


# **Créer la matrice des prédicteurs `X`.** Indice: `X` doit avoir 2 colonnes...

# In[37]:


# Compléter le code ci-dessous ~ 3 lignes
NBA['x0']= np.ones(m)
X = NBA[['x0', 'PTSdiff']].values

x0 = np.ones((m))
x1 = NBA['PTSdiff']
#X = np.array((x0,x1))


# In[50]:





# **Vérifier la dimension de la matrice des prédicteurs `X`. Quelle est la dimension de `X` ?**

# In[38]:


# Compléter le code ci-dessous ~ 1 ligne
X.shape


# **Créer le modèle de référence (baseline)**

# In[39]:


# Compléter le code ci-dessous ~ 1 ligne
## le plus simple possible
y_baseline = y.mean()


# **À l'aide de l'équation normale, trouver les paramètres optimaux du modèle de régression linéaire simple**

# In[40]:


# Compléter le code ci-dessous ~ 1 ligne
theta = np.dot(np.dot(np.linalg.pinv(np.dot(X, X.T)), X).T, y)
theta


# **Calculer la somme des carrées des erreurs (SSE)**

# In[41]:


### Sum of squared errors
# Compléter le code ci-dessous ~ 1 ligne
error = np.dot(theta.T, X.T) - y  # (theta_0 + theta_1*x) - y
SSE = np.dot(error, error.T) # la somme de carrés
SSE


# In[42]:


error


# In[43]:


theta


# **Calculer la racine carrée de l'erreur quadratique moyenne (RMSE)**

# In[44]:


### RMSE = Root Mean Square Error
# Compléter le code ci-dessous ~ 1 ligne
RMSE = np.sqrt(SSE/m)
RMSE


# **Calculer le coefficient de détermination $R^2$**

# In[45]:


y_baseline


# In[46]:


### Une valeur près de 1 = bonne
# Compléter le code ci-dessous ~ 1-2 lignes
# R2 -1 - (SSE/Baseline SSE)
# y_baseline -> erreur par rapport au baseline
R2 = 1-SSE/np.sum((y-y_baseline)**2)
R2 


# **Affichage des résultats**

# In[47]:


fig, ax = plt.subplots()
ax.scatter(x1, y,label="Data points")
reg_x = np.linspace(-1000,1000,50)
reg_y = theta[0] + np.linspace(-1000,1000,50)* theta[1]
ax.plot(reg_x, np.repeat(y_baseline,50), color='#777777', label="Baseline", lw=2)
ax.plot(reg_x, reg_y, color="g", lw=2, label="Modèle")
ax.set_xlabel("Différence de points", fontsize=16)
ax.set_ylabel("Nombre de victoires", fontsize=16)
ax.legend(loc='upper left', fontsize=16)


# ### 3 - Régression linéaire multiple

# Nous allons maintenant tenter de prédire le nombre de points obtenus par une équipe donnée au cours de la saison régulière en fonction des autres variables explicatives disponibles. Nous allons mettre en oeuvre plusieurs modèles de régression linéaire multiple

# **Stocker le nombre de points marqués au cours de la saison dans la variable `y`. Il s'agira de la varible que l'on cherche à prédire**

# In[63]:


# Compléter le code ci-dessous ~ 1 ligne
y = NBA['PTS'].values
y


# **Créer la matrice des prédicteurs `X` à partir des variables `2PA` et `3PA`**

# In[66]:


# Compléter le code ci-dessous ~ 3 lignes
NBA['x0']= np.ones(m)
X = NBA[['x0', '2PA', '3PA']].values.T
X
#X3p = NBA[['x0', '3PA']].values

#x0 = np.ones((m))
#x1 = NBA['PTSdiff']
#X = np.array((x0,x1))


# In[67]:


# Compléter le code ci-dessous ~ 1 ligne
X.shape


# **Vérifier la dimension de la matrice des prédicteurs `X`. Quelle est la dimension de `X` ?**

# **Créer le modèle de référence (baseline)**

# In[68]:


# Compléter le code ci-dessous ~ 1 ligne
y_baseline = y.mean()
y_baseline


# **À l'aide de l'équation normale, trouver les paramètres optimaux du modèle de régression linéaire**

# In[69]:


# Compléter le code ci-dessous ~ 1 ligne
A = np.dot(X, X.T)
A_inv = np.linalg.inv(A)
theta = np.dot(np.dot(A_inv, X), y)
theta 


# **Calculer la somme des carrées des erreurs (SSE)**

# In[75]:


# Compléter le code ci-dessous ~ 1 ligne
error = np.dot(theta, X)-y
SSE = np.dot(error, error.T)


# **Calculer la racine carrée de l'erreur quadratique moyenne (RMSE)**

# In[76]:


# Compléter le code ci-dessous ~ 1 ligne
MSE = SSE/m
RMSE = np.sqrt(MSE)


# **Calculer le coefficient de détermination $R^2$**

# In[77]:


# Compléter le code ci-dessous ~ 1-2 lignes
bas_err = y_baseline -y
R2 = 1 - SSE/np.dot(bas_err.T, bas_err)
R2


# ### 3 - Ajouter les variables explicatives FTA et AST

# **Recommencer les étapes ci-dessus en incluant les variables FTA et AST**

# In[ ]:


None


# ### 4 - Ajouter les variables explicatives ORB et STL

# **Recommencer les étapes ci-dessus en incluant les variables ORB et STL**

# In[ ]:


None


# ### 5 - Ajouter les variables explicatives DRB et BLK

# **Recommencer les étapes ci-dessus en incluant les variables DRB et BLK**

# In[ ]:


None


# ### 6 - Optionnel - Regression polynomiale

# Ajouter des variables explicatives de type polynomiales

# ### Fin du TP
