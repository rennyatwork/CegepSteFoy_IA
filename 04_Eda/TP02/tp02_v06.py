#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as stm
import statsmodels.stats.weightstats as ws
import sklearn.linear_model as sk_lin_mod
import math
import seaborn as sns
import os


# In[3]:


get_ipython().run_line_magic('matplotlib', '')


# In[4]:


os.getcwd()


# In[5]:


print (os.getcwd())
currDir = os.getcwd()
fullPath = currDir + "/CabaneASucrev0r3.csv"
type(currDir)


# In[6]:


donnee = pd.read_csv(fullPath)
stats=donnee.describe()
dimensions=donnee.shape
stats


# In[7]:


donnee.isnull().sum()


# Heureusement, très peu de nulls.
# 
# Quels sont les enregistrements contenant des nulls?
# 

# In[8]:


donnee[(donnee['Temp max.(°C)'].isna()==True) | (donnee['Précip. tot. (mm)'].isna()==True)] 


# Fill na

# In[9]:


donnee.fillna(method='bfill', inplace=True)
donnee.isnull().sum()


# In[10]:


donnee.columns


# Supprimer les colonnes pixel

# In[11]:


donnee.filter(like='Pixel').columns
dfSansPixel = donnee.drop(donnee.filter(like='Pixel').columns, axis=1)
dfSansPixel.columns


# In[12]:


dfSansPixel.dtypes


# In[13]:


### liste de variables (colonnes) dépendantes
col_debit_seve = dfSansPixel['Débit sève (L/j)']
col_sucre_dans_seve = dfSansPixel['Sucre sève (%)']
col_pct_transmittance = dfSansPixel['Transmittance produit (%)']
col_productivite_seve_par_saison = dfSansPixel['Production moyenne par entaille (L)']
list_cols_dependennt_vars = [col_debit_seve, col_sucre_dans_seve, col_pct_transmittance, col_productivite_seve_par_saison]


# In[14]:


## convertir Classe Sirop en numérique - https://www.youtube.com/watch?v=wH_ezgftiy0&t=136s
dfSansPixel['CategClasseSirop'] = dfSansPixel['Classe Sirop'].astype('category').cat.codes
dfNumerique = dfSansPixel.select_dtypes(exclude='object').copy()
dfNumerique.dtypes


# ### Multiple Linear Regression
# 
# 
# https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/
# 
# Assumption of Regression Model : 
# 
# - Linearity: The relationship between dependent and independent variables should be linear.
# - Homoscedasticity: Constant variance of the errors should be maintained.
# - Multivariate normality: Multiple Regression assumes that the residuals are normally distributed.
# - Lack of Multicollinearity: It is assumed that there is little or no multicollinearity in

# In[15]:


stats = dfNumerique.describe()
stats


# In[16]:


#dfNumerique[dfNumerique['CategClasseSirop']==0 | dfNumerique['Pression osmoseur (bar)']==0].count()
dfNumerique[dfNumerique['Pression osmoseur (bar)']==0].count()


# In[17]:


dfNumerique = dfNumerique.loc[dfNumerique['Pression osmoseur (bar)'] >0]


# ### Colinearity

# In[18]:


colsX = dfNumerique.loc[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])]


# In[19]:


## https://www.geeksforgeeks.org/multicollinearity-in-data/

# calculate the variance inflation factor
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
def calculate_vif( pXcols):
    # compare with each columns
    for i in range(len(pXcols)):
        vif_list = [vif(pXcols.values, index) for index in range(len(pXcols))]
        maxvif = max(vif_list)
        print("Max VIF value is ", maxvif)                     
        drop_index = vif_list.index(maxvif)
        print("For Independent variable", pXcols[drop_index])
    
        if maxvif > 10:
        
            print("Deleting", pXcols[drop_index])
            #pXcols = pXcols.delete(drop_index)
            print("Final Independent_variables ", pXcols)


# In[20]:


colsX.columns


# In[21]:


calculate_vif(colsX.columns)


# In[ ]:


x=1


# In[ ]:


## sns tips
#sns.pairplot(dfNumerique)


# In[3]:


## pour chaque var dépendant, une régression
#for col_y in list_cols_dependennt_vars:
#    sns.pairplot(dfNumerique, y_vars=col_y.name, kind='reg')


# In[ ]:


corr = dfNumerique.corr()
#sns.heatmap(corr, cbar=True, cmap="Blues", center=0, annot=True, fmt=".1f")

## le résultat est une matric 22x22 pas facile à visualiser


# In[ ]:


corr


# In[24]:


## On voit qu'il y a certaines correlations parfaites
## ex: (temp moyen, temp max), (temp moyen, temp min), (moyenne entaille, episode gel/degel)
## (pression osmoseur bar, boullioire 0c), (pression osmoseur bar, quantité sirop obtenu %)
## (osmoseur heures opération, alimentation osmoseur (L/j))
## (Sucre sortie osmoseur (%), pression osmoseur)
## (alimentation osmoseur, (L/j), temps boulloire) -> 0.89
## (débit sève, heures opération/j)
## Enlevons quelques unes de ces variables et revoyons la correlation
list_col_redondantes = ['Temp min.(°C)', 'Temp max.(°C)', 'Température Bouilloire (0C)'
                       , 'Quantité de sirop obtenue (L)', 'Sucre du sirop obtenu (%)'
                       , 'Osmoseur (heures opération/j)', 'Sucre sortie osmoseur (%)'
                       , 'Temps bouilloire (h)'
                       ]

## On repète l'opération antérieur avec moins de colonnes
dfNumerique = dfNumerique.loc[:, ~dfNumerique.columns.isin(list_col_redondantes)]

corr = dfNumerique.corr()

## heatmap sans correlations parfaites ou presque parfaite:
sns.heatmap(corr, cbar=True, cmap="Blues", center=0, annot=True, fmt=".2f")


# ### At this point, we have strong relationships:
# - Débit sève (L/j) --> Alimentation osmoseur (L/j) [0.97]
# - Sucre sève (%) --> Pression osmoseur (bar) [0.9]
# - Transmittance produit (%) --> Pression osmoseur (bar)[0.81], Sucre sortie osmoseur (%) [0.68]
# - Production moyenne par entaille (L) --> Nombre épisodes gel/dégel [0.95]

# In[23]:


## Transmitance
### pression_osmoseur_vs_transmittance
sns.lmplot(x='Pression osmoseur (bar)', y = 'Transmittance produit (%)', data = dfNumerique)


# In[ ]:


## Le graphique ression_osmoseur_vs_transmittance montre une concentration
## dans x près de 40. Regardons s'il y a des outliers:
sns.boxplot(x='Pression osmoseur (bar)',data = dfNumerique)


# In[20]:


## effectivement, le boxplot nous montre la présence des outliers
stats = dfNumerique['Pression osmoseur (bar)'].describe()
stats


# In[22]:


### regardons combien de 0:
sns.histplot(x='Pression osmoseur (bar)',data = dfNumerique)


# In[16]:


#sns.pairplot(dfNumerique)


# In[32]:


## correlation seulement entre les variables indépendantes

dfDependantVars= dfNumerique.loc[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])]
type(dfDependantVars)

corr2 = dfDependantVars.corr()
sns.heatmap(corr2, cbar=True, annot=True, cmap="Blues", fmt=".02f", center=0)


# Multiple Linear Regression - https://www.youtube.com/watch?v=J_LnPL3Qg70
# 

# In[16]:


from sklearn import linear_model
reg = linear_model.LinearRegression()

## https://www.statology.org/pandas-exclude-column/
#select all columns except 'rebounds' and 'assists'
#df.loc[:, ~df.columns.isin(['rebounds', 'assists'])]
list_reg = []
for col in list_cols_dependennt_vars:
 #   print(col.name)
    #result = reg.fit(dfNumerique[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])], dfNumerique[col.name])
    result = reg.fit(dfNumerique.loc[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])], dfNumerique[col.name])
    list_reg.append(result)


#dfNumerique['Année'].name
#list_cols_dependennt_vars[0].name
#[col.name for col in list_cols_dependennt_vars]
#reg.fit(dfNumerique.loc[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])], dfNumerique['Débit sève (L/j)'])
#type(dfNumerique.loc[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])])
#dfNumeriqueIndepVar = dfNumerique[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])]


# In[24]:


print("coefs: ", list_reg[0].coef_.round(2))
print("intercept: ", list_reg[0].intercept_)


# fonction vérif distr normale

# In[17]:


def print_dist_norm(pCol, pXlabel, pYlabel="Fonction de densité f(x)", pNbRuns=1000):
    grille_x = np.linspace(pCol.min(), pCol.max(), pNbRuns)
    dx=(pCol.max()-(pCol.min()))/(pNbRuns-1)
    mu, sigma = sts.norm.fit(pCol.values)
    param=sts.norm.fit(pCol.values)
    pdf = sts.norm.pdf(grille_x, mu, sigma)
    ax=pCol.plot.hist(density=True, bins = 10, color = 'blue', edgecolor = 'black')
    ax.set_xlabel(pXlabel)
    ax.plot(grille_x, pdf, linewidth=3, color = 'red')
    ax.set_ylabel(pYlabel)

