# %%
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
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy import stats
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression

# %%
#import sys
#!{sys.executable} -m pip install --upgrade ipykernel
#!{sys.executable} -m pip install --upgrade pandas
#!{sys.executable} -m pip install --upgrade  sickit-learn 
#!{sys.executable} -m pip install featurewiz

# %%
import PyQt5 #works
from PyQt5 import QtCore, QtGui, QtWidgets #works for pyqt5
%matplotlib 

# %%
os.getcwd()

# %%
print (os.getcwd())
currDir = os.getcwd()
fullPath = currDir + "/CabaneASucrev0r3.csv"
type(currDir)

# %%
donnee = pd.read_csv(fullPath)
stats=donnee.describe()
dimensions=donnee.shape
stats

# %%
donnee.isnull().sum()

# %%
"""
Heureusement, très peu de nulls.

Quels sont les enregistrements contenant des nulls?

"""

# %%
donnee[(donnee['Temp max.(°C)'].isna()==True) | (donnee['Précip. tot. (mm)'].isna()==True)] 

# %%
"""
Fill na
"""

# %%
donnee.fillna(method='bfill', inplace=True)
donnee.isnull().sum()

# %%
donnee.columns

# %%
"""
Supprimer les colonnes pixel
"""

# %%
donnee.filter(like='Pixel').columns
dfSansPixel = donnee.drop(donnee.filter(like='Pixel').columns, axis=1)
dfSansPixel.columns

# %%
dfSansPixel.dtypes

# %%
dfSansPixel.describe().apply(lambda x: round(x, 2))

# %%
### liste de variables (colonnes) dépendantes
list_cols_dependennt_vars = []
col_debit_seve = dfSansPixel['Débit sève (L/j)'].name
col_sucre_dans_seve = dfSansPixel['Sucre sève (%)'].name
col_pct_transmittance = dfSansPixel['Transmittance produit (%)'].name
col_productivite_seve_par_saison = dfSansPixel['Production moyenne par entaille (L)'].name
list_cols_dependennt_vars = [col_debit_seve, col_sucre_dans_seve, col_pct_transmittance, col_productivite_seve_par_saison]

# %%
## convertir Classe Sirop en numérique - https://www.youtube.com/watch?v=wH_ezgftiy0&t=136s
#dfSansPixel['CategClasseSirop'] = dfSansPixel['Classe Sirop'].astype('category').cat.codes
dfNumerique = dfSansPixel.select_dtypes(exclude='object').copy()
dfNumerique.describe().apply(lambda x: round(x, 2))

# %%
### convertir Classe Sirop en var numérique
###
# Creating numeric columns
###
colsCategSirop = pd.get_dummies(dfSansPixel['Classe Sirop'], prefix='categSirop_', drop_first=True)

## concatenate side by side
dfNumerique = pd.concat([dfNumerique, colsCategSirop], axis=1)
#dfNumeriue.columns


# %%
#### enelever les données où débit nul de sève
#dfNumerique = dfNumerique.loc[dfNumerique['Pression osmoseur (bar)'] !=0]
dfNumerique = dfNumerique.loc[dfNumerique['Débit sève (L/j)'] !=0]

# %%
dfNumerique.columns

# %%
dfSansPixel[dfSansPixel['Classe Sirop']=='0'].count()

# %%
### Standardization
## https://datagy.io/pandas-normalize-column/
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(dfNumerique)
scaled = scaler.fit_transform(dfNumerique)
dfNumeriqueStd = pd.DataFrame(scaled, columns=dfNumerique.columns)

# %%
## book Practical Statiscs for Data Scientists, cahp 2
## https://github.com/gedeck/practical-statistics-for-data-scientists/blob/master/python/notebooks/Chapter%202%20-%20Data%20and%20sampling%20distributions.ipynb
def plot_prob_var_y(pDf):
    for y in list_cols_dependennt_vars:
        col_y = pDf[y][pDf[y]>0]
        print("y", y)
        #print("col[y]", dfNumerique[y][dfNumerique[y]>0])
        np.diff(np.log(col_y))
    
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title( y)
        sts.probplot(col_y, plot=ax)

# %%
#plot_distr_var_y(dfNumerique)

# %%
#plot_distr_var_y(dfNumeriqueStd)

# %%
dfNumerique.describe().apply(lambda x: round(x, 2))

# %%
dfNumeriqueStd.describe().apply(lambda x: round(x, 2))

# %%
def plot_box_plot_vars_y(pDf):
    for y in list_cols_dependennt_vars:
        col_y = pDf[y][pDf[y]>0]
        print("y", y)
    
        plt.figure()
        sns.boxplot(y=col_y).set_title(y)
        plt.show(block=False)

# %%
plot_box_plot_vars_y(dfNumerique)

# %%


# %%
len(dfNumerique)

# %%
"""
---
Si on enlève seulement les outliers Débit de sève', on enlève plus que 1/3 du dataset.
Décision: on laisse tel que

---
"""

# %%
"""
### Multiple Linear Regression


https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/

Assumption of Regression Model : 

- Linearity: The relationship between dependent and independent variables should be linear.
- Homoscedasticity: Constant variance of the errors should be maintained.
- Multivariate normality: Multiple Regression assumes that the residuals are normally distributed.
- Lack of Multicollinearity: It is assumed that there is little or no multicollinearity in
"""

# %%
stats = dfNumerique.describe()
stats

# %%
dfNumerique['Temps bouilloire (h)'].sort_values(kind='quicksort')

# %%
dfNumerique['Temps bouilloire (h)'].describe()

# %%
### col 'Temps bouilloire (h)'
### 4.8 aprox mean + 1.5 std
outlier = (dfNumerique['Temps bouilloire (h)'].mean() +1.5*dfNumerique['Temps bouilloire (h)'].std())
dfNumerique['Temps bouilloire (h)'][dfNumerique['Temps bouilloire (h)']>outlier]

# %%
dfNumerique = dfNumerique.loc[dfNumerique['Temps bouilloire (h)']<outlier]
dfNumerique.describe()

# %%
#### col osmoseur heures opérations/j
#dfNumerique['Osmoseur (heures opération/j)'].describe()

# %%
#outlier = (dfNumerique['Osmoseur (heures opération/j)'].mean() +1.5*dfNumerique['Osmoseur (heures opération/j)'].std())
#print(outlier)
#dfNumerique['Osmoseur (heures opération/j)'][dfNumerique['Osmoseur (heures opération/j)']]>outlier]

# %%
#dfNumerique = dfNumerique.loc[dfNumerique['Osmoseur (heures opération/j)']<outlier]


# %%
#dfNumerique.describe()

# %%
corr = dfNumerique.corr()
#sns.heatmap(corr, cbar=True, cmap="Blues", center=0, annot=True, fmt=".1f")

## le résultat est une matric 22x22 pas facile à visualiser

# %%
#corr['Année'].sort_values(ascending=False)
corr[(corr['Année']<1) & (corr['Année']>0.2)]['Année'].sort_values(ascending=False)

# %%
sorted_pairs = corr.unstack().sort_values(kind="quicksort", ascending=False).dropna()
len(sorted_pairs)

# %%
x = sorted_pairs.where((sorted_pairs <1.0) & (sorted_pairs >0) ).dropna()
list_corr = sorted_pairs.tolist()
lst = [l for l in list_corr if (l < 1) & (l> 0) & (not(math.isnan(l)))]

# %%
type(corr.unstack())

# %%


# %%
#sns.heatmap(corr, cbar=True, center=0, annot=True, fmt="0.2f", cmap="Blues")

# %%
"""
### Séparation des variables dépendantes (Y) et indépendantes (X)
"""

# %%
## seulement les variables X
dfNumerique_X_cols = dfNumerique.loc[:, ~dfNumerique.columns.isin(list_cols_dependennt_vars)].copy()

lstColTemp = ['Temp max.(°C)', 'Temp min.(°C)']
dfNumerique_X_cols = dfNumerique_X_cols.loc[:,~dfNumerique_X_cols.columns.isin(lstColTemp) ].copy()

## seulement les variables Y
dfNumerique_Y_cols = dfNumerique.loc[:, dfNumerique.columns.isin(list_cols_dependennt_vars)].copy()

len(dfNumerique.columns)

# %%
#### Après coup, on a appris que ces 2 variables causaient un prob de haute colinéarité.
#### et si on les enlève plus tôt?

lstColsExlude = ['Précip. Tot. Hiver (mm)', 'Temp moy.(°C)']
dfNumerique_X_cols = dfNumerique_X_cols.loc[:, ~dfNumerique_X_cols.columns.isin([col for col in lstColsExlude])]

# %%
"""
#### Evaluate model
"""

# %%
"""
=============== Evaluating model ===========

https://www.youtube.com/watch?v=VCVhwjbI6h8

========
"""

# %%
### Fonction qui imprime le sommaire du modèle
def print_model_summary(pDfColsX, pDfColsY):
    ## Add constant
    pDfColsX = stm.add_constant(pDfColsX)
    lst_models = []
    for y in (pDfColsY.columns.values):
    
        col_y = dfNumerique_Y_cols[y]
    
        model = stm.OLS(col_y, pDfColsX.assign(const=1)).fit()
        print("=========")
        print("var depend: ", y)
    
        #lst_models.append(model)
        print(model.summary())
        #sns.distplot(model.resid, fit=sts.norm)        
        #p = sns.histplot(model.resid, kde=True, stat="density").set(title = y)
        

# %%
print("---- Sommaire dfNumerique ----")
print_model_summary(dfNumerique_X_cols, dfNumerique_Y_cols)

# %%


# %%
### https://www.youtube.com/watch?v=VCVhwjbI6h8
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# %%
def analyze_model(pColsY, pDfDependentVars):
    for y in (pColsY.columns.values):    
        col_y = pColsY[y]
        X_train, X_test, y_train, y_test = \
        train_test_split(pDfDependentVars, col_y, test_size=0.25, random_state=0)


        ## transforming data    
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)


        ## Fitting Multiple Linear Regression to the training set
        regressor =  LinearRegression()
        regressor.fit(X_train, y_train)
        

        y_pred = regressor.predict(X_test)
        mse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        #r_sq = regressor.score(X_test, y_test)

        ##adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
        ## Where n is the sample size and p is the number of independent variables.

        #print ("len(X_test): ", len(X_test))
        #print ("len(y_test): ", len(y_test))
    
        ### ajd_r2
        ## https://stackoverflow.com/questions/51038820/how-to-calculated-the-adjusted-r2-value-using-scikit
        ## https://www.dummies.com/article/business-careers-money/business/accounting/calculation-analysis/how-to-calculate-the-adjusted-coefficient-of-determination-146054/
        #print ("n: ", n)
        #print ("p: ", p)
        #print ("adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)")

        n = len(X_test)
        p = len(pDfDependentVars.columns)
        adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
        print("--------")
        
        print("[col_y]: ", col_y.name)
        print("[mse]: ", mse)
        #print("[r2] - A value of 1 indicates that the response variable can be perfectly explained by the predictor variables.")
        print("[r2]: ", r2)
        print("[adj_r2]: ", adj_r2)
        print("[coef]: ", regressor.coef_)
        print("[intercept_]: ", regressor.intercept_)
        #print("[r_sq]: ", r_sq)

# %%
analyze_model(dfNumerique_Y_cols, dfNumerique_X_cols)

# %%
"""
\---

Sans rien faire, on a un modèle très précis (r2 et adj_r2 élevés).
Mais pourrait-on avoir un bon résultat avec moins de colonnes?

\---
"""

# %%
### https://github.com/simaria22/prediction_heart_failure/blob/master/feature_select.py
## Ajoute des colonnes au modèle pendant que la nouvelle colonne améliore
## le modèle OU
## jusqu'à qu'un certain niveau de précision soit atteint
def get_reduced_df_X(pDfx, pY, pAccuracy=0.7):
    print("------")
    print("dependent var y: ", pY.name)
    
    # new X df with fewer columns than pDfx
    reducedDf = []

    # column names to keep
    dfKeeper = None

    #no of features
    nof_list=np.arange(1,len(pDfx.columns))            
    high_score=0
    #Variable to store the optimum features
    nof=0           
    score_list =[]
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(pDfx, pY, test_size = 0.3, random_state = 0)
        model = LinearRegression()
        rfe = RFE(model,n_features_to_select= nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        
        print("n: ", n)
        print("score: ", score)
        print("high_score: ", high_score)

        # print summaries for the selection of attributes
        print("rfe.support_: ",rfe.support_)
        print("rfe.ranking_: " ,rfe.ranking_)
        
        #dfKeeper = pd.DataFrame(rfe.support_,index=pDfx.columns,columns=['Rank'])
        data= {'colName': pDfx.columns, 'keep': rfe.support_ }
        dfKeeper = pd.DataFrame(data)
        dfKeeper = dfKeeper[dfKeeper["keep"]==True]
        if(score>high_score):
            high_score = score
            nof = nof_list[n]
        if (score > pAccuracy):
            break

    
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    print("df columns to keep: ")
    #print("shape")
    #print(dfNumerique_X_cols[dfKeeper["colName"]].shape)
    print(dfKeeper)
    
    reducedDf = dfNumerique_X_cols[dfKeeper["colName"]]
    
    #print("reducedDf: ", reducedDf.head())
    #print("unique: ", reducedDf[dfKeeper["colName"]].unique())
    return reducedDf

# %%
### création d'un dictionnaire contenant la variable dépendantes et ses variables indépendantes
### correspondantes
### clé = colonne dépendante, valeur = dataframe des colonnes avec haute influence
dict_y_colsX = {}

for y in (dfNumerique_Y_cols.columns.values):    
    col_y = dfNumerique_Y_cols[y]
    reducedDf = get_reduced_df_X(dfNumerique_X_cols, col_y)
    #print("reduceDf.columns.values: ",  reducedDf.columns.values)
    #print("unique values: ", reducedDf[reducedDf.columns.values[0]].unique())
    #print("len: ", len(reducedDf[reducedDf.columns.values[0]]))
    #print("Nan: ", reducedDf[reducedDf.columns.values[0]].isnull().sum())
    ## call to analyze_model
    #analyze_model(dfNumerique_Y_cols, reducedDf)
    #analyze_model(col_y.to_frame(), reducedDf)
    dict_y_colsX.update({col_y.name: reducedDf})

# %%
### analyse le modèle de nouveau avec un dataframe réduit
#analyze_model(dfNumerique_Y_cols, dfNumerique_X_cols)
for key in (dict_y_colsX.keys()):
    print("key: ", key)
    dfX = dict_y_colsX.get(key)
    dfY = dfNumerique_Y_cols[key]
    #print(type(dfX))
    #print(type(dfY))
    #dfX =pd.DataFrame(dict_y_colsX[key], columns=[key])
    #print("shape: ", dfX.shape)
    #print("nulls: ", dfX.isnull().sum())
    analyze_model(dfY.to_frame(), dfX)
    

# %%
### analyse le modèle de nouveau avec un dataframe réduit
#analyze_model(dfNumerique_Y_cols, dfNumerique_X_cols)
for key in (dict_y_colsX.keys()):
    print("key: ", key)
    dfX = dict_y_colsX.get(key)
    dfY = dfNumerique_Y_cols[key]    
    print_model_summary(dfX, dfY.to_frame())

# %%
"""
-----

À ce point, on a un modèle avec une bonne précision (R-squared et Adj R-squared) pour 
toutes les variables dépendantes.

Seuelement la variable 'Transmittance produit (%)' contient un advertissement de colinéarité

-----
"""

# %%
#### outliers
### book: Practical statistcs for data scientists, pg 178
## https://github.com/gedeck/practical-statistics-for-data-scientists
dict_outliers = {}
for key in (dict_y_colsX.keys()):
    print("--------------------------")
    print("key: ", key)
    dfX = dict_y_colsX.get(key)
    dfY = dfNumerique_Y_cols[key]    
    outlier = stm.OLS(dfY, dfX.assign(const=1))
    result = outlier.fit()
    influence = OLSInfluence(result)
    sresiduals = influence.resid_studentized_internal
    sresiduals.idxmin(), sresiduals.min()
    outlier = dfNumerique.loc[sresiduals.idxmin(), :]
    print("[outler[key]: ", outlier[key])
    print("[outlier values]: ", outlier[dfX.columns.values])

# %%
### pairplots
#analyze_model(dfNumerique_Y_cols, dfNumerique_X_cols)
i=0
for key in (dict_y_colsX.keys()):
    i+=1
    print("key: ", key)
    dfX = dict_y_colsX.get(key)
    dfY = dfNumerique_Y_cols[key]    
    #sns.pairplot(pd.concat([dfX, dfY], axis=0))
    #plt.figure(i)
    #print(dfX.columns.values)
    #print(dfNumerique[key].head())
    #plt.figure()
    #plt.plot(pd.concat([dfX, dfY], axis=0))
    sns.pairplot(data = dfNumerique, x_vars = dfX.columns.values , y_vars = [key], kind='reg', diag_kind=None)
    
    #sns.pairplot(data = reducedDf, x_vars = ['Nombre épisodes gel/dégel'], y_vars = ['Production moyenne par entaille (L)'], kind='reg', diag_kind='kde')
    #plt.show(block=False)

    

# %%
### plot regressions
#https://seaborn.pydata.org/tutorial/regression.html
i=0
for key in (dict_y_colsX.keys()):
    
    print("key: ", key)
    dfX = dict_y_colsX.get(key)
    dfY = dfNumerique_Y_cols[key]    
    for col_x in dfX.columns.values:
        i+=1
        plt.figure(i)
        print("col_x: ", col_x)
        sns.regplot(data = dfNumerique, x = col_x, y = key)    
        plt.show(block=False)

# %%
i=0
for key in (dict_y_colsX.keys()):
    
    print("key: ", key)
    dfX = dict_y_colsX.get(key)
    dfY = dfNumerique_Y_cols[key]    
    for col_x in dfX.columns.values:
        i+=1
        plt.figure(i)
        print("col_x: ", col_x)
        #sns.boxplot(x=dfX[col_x])  
        sns.histplot(data=dfX, bins=10, x=col_x)
        plt.show(block=False)

#i=0    
#for col_x in dfX.columns:
#    i+=1
#    plt.figure(i)
    #print("col_x: ", col_x)
    #print(type(col_x))
 #   sns.boxplot(x=col_x.to_frame())    
 #   plt.show(block=False)

# %%
### boxplot vars y
#https://seaborn.pydata.org/tutorial/regression.html
i=0
for key in (dict_y_colsX.keys()):    
    print("key: ", key)
    dfX = dict_y_colsX.get(key)
    dfY = dfNumerique_Y_cols[key]    
    i+=1
    plt.figure(i)    
    sns.histplot(data=dfY, bins=10)
    plt.show(block=False)

# %%
"""
------

la var dépendante Transmittance produit (%) a un problème de multicolinearité 

------
"""

# %%
#### END Evaluate model

# %%
##### Vif - régler la colinéarité de la variable

# %%
### vif
## https://www.kdnuggets.com/2019/07/check-quality-regression-model-python.html
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# %%
### https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = dfNumerique_X_cols.columns


# %%

# calculating VIF for each feature

vif_data["VIF"] = [vif(dfNumerique_X_cols.values, i)
                          for i in range(len(dfNumerique_X_cols.columns))]

vif_data.sort_values(by="VIF", axis=0, kind="quicksort",  ascending=False)


# %%
dfVifPlusGrand10 = vif_data[vif_data["VIF"] >10]
len(dfVifPlusGrand10)

# %%
### dfNumerique_X sans les vif >10
dfNumerique_X_cols_vif =[]
dfNumerique_X_cols_vif = dfNumerique_X_cols.loc[:, ~dfNumerique_X_cols.columns.isin(dfVifPlusGrand10["feature"].values)].copy()
#for x in (dfVifPlusGrand10["feature"].values):
#    print("col: ", x)
dfNumerique_X_cols_vif.columns

# %%
sns.heatmap(dfNumerique_X_cols_vif.corr(), cbar=True, cmap="Blues", center=0, annot=True, fmt=".2f")

# %%
"""
\-------------------\
Est-ce que l'on obtien un résultat meilleur pour 
'Transmittance produit (%)' ?\
\----------------\
"""

# %%
### appelle get_reduced_df_X avec 
### X =  dfNumerique_X_cols_vif et Y = dfNumerique_Y_cols['Transmittance produit (%)']
newDfTransmittance = get_reduced_df_X(dfNumerique_X_cols_vif, dfNumerique_Y_cols['Transmittance produit (%)'])
print_model_summary(newDfTransmittance, dfNumerique_Y_cols['Transmittance produit (%)'].to_frame())

# %%
analyze_model( dfNumerique_Y_cols['Transmittance produit (%)'].to_frame(), newDfTransmittance)

# %%
### Le résultat n'est pas meilleur
## éliminons 'Précip. Tot. Hiver (mm)' et Temp moy.(°C) à cause de leur haute correlation
## ('Précip. Tot. Hiver (mm)', 'Année') = 0.78 et ('Temp moy.(°C)', 'Calendrier saison') = 0.73


### appelle get_reduced_df_X avec 
### X =  dfNumerique_X_cols_vif et Y = dfNumerique_Y_cols['Transmittance produit (%)']
#dfDependantVars= dfNumerique.loc[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])]
lstColsExlude = ['Précip. Tot. Hiver (mm)', 'Temp moy.(°C)']
dfNumerique_X_cols_vif = dfNumerique_X_cols_vif.loc[:, ~dfNumerique_X_cols_vif.columns.isin([col for col in lstColsExlude])]
newDfTransmittance = get_reduced_df_X(dfNumerique_X_cols_vif, dfNumerique_Y_cols['Transmittance produit (%)'])
print_model_summary(newDfTransmittance, dfNumerique_Y_cols['Transmittance produit (%)'].to_frame())

# %%
####
###  Fnalement on n'a plus le problème de colinéarité
###

# %%
#### Réponses aux questions 1
### Les tableaux ci-bas donnent, entre autres les informations concernant le adj. r-squared
###, t-value, p-value
### et coefficients des modèles.
### !!!NOTE!!! générer les graphiques et le sauvegarder à part!!!

# %%
"""

"""

# %%
"""

-----------------------------
"""

# %%
"""
#######
"""

# %%
"""
\---------------

Q2)
Hypothesis testing

\------------
"""

# %%
"""
\--------------\

Vars:
'Transmittance produit (%)'
'Sucre sève (%)'


Y-at-il une variance significative entre les années 2014-2016?


Obtenir la moyenne pour les année 2014 à 2016 e comparer les résultats



\---------------\
"""

# %%
### obtenir moyenne d'une colonne
def getMeanListExclusions(pColName, pData=dfNumerique, pListeExclusion=[2014, 2015, 2016]):
    return pData[pColName][~pData['Année'].isin(pListeExclusion)].mean()

# %%
#moyenne_transmit_toutes_annees = dfNumerique['Transmittance produit (%)'].mean()
moyenne_transmit_toutes_annees = getMeanListExclusions('Transmittance produit (%)')
moyenne_sucre_toutes_annees = getMeanListExclusions('Sucre sève (%)')

# %%
moyenne_transmit_toutes_annees

# %%
moyenne_sucre_toutes_annees

# %%
"""
\----------------\

Quelles sont les moyennes pour les années de 2014 à 2016?

\----------------\
"""

# %%
### obtenir moyenne d'une colonne
def getMeanListInclusions(pColName, pAnnee, pData=dfNumerique ):
    return pData[pColName][pData['Année'] == pAnnee].mean()

# %%
### for these dictionnaries, key = year [2014-2016]
lstAnnees = [2014, 2015, 2016]
dictAnneeTransmittance = {}
dictAnneeSucre = {}
for annee in lstAnnees:    
    dictAnneeTransmittance.update({annee: getMeanListInclusions('Transmittance produit (%)', annee)})
    dictAnneeSucre.update({annee: getMeanListInclusions('Sucre sève (%)', annee)})

# %%
dictAnneeTransmittance

# %%
dictAnneeSucre

# %%
### Imnprime la différence entre la moyenne historique (excluant 2014-2016) versus [2014-2016]
def compare_means(pMean1, pMean2, pAnnee, pCol):
    diff = pMean1 - pMean2
    print("-----")
    print("Différence entre moyennes pour ", pCol)
    print("moyenne historique: ", pMean1)
    print("moyenne ", pAnnee)
    print("Différence absolue: ", abs(diff))
    print("Différence %: ", abs(100*(diff/pMean1)))
    

# %%
### Transmittance
for key in dictAnneeTransmittance.keys():
    compare_means(moyenne_transmit_toutes_annees, dictAnneeTransmittance.get(key), key, 'Transmittance produit (%)')

# %%
### Transmittance
for key in dictAnneeSucre.keys():
    compare_means(moyenne_sucre_toutes_annees, dictAnneeSucre.get(key), key, 'Sucre sève (%)')

# %%
## https://towardsdatascience.com/demystifying-hypothesis-testing-with-simple-python-examples-4997ad3c5294
def print_analyse_hypothese(pColYName, pMoyenneHistoriqueY, pDict, pDf=dfNumerique, pConf=0.05):
    ## Transmittance produit (%)
    n =len(pDf)
    pnull = pMoyenneHistoriqueY ##0.43732148760330574 #moyenne hystorique
    ### Transmittance
    print("==================================")
    print("var [pColYName]: ", pColYName)
    for key in pDict.keys():
        phat = pDict.get(key) #moyenne de l'année
        print("-------------------")
        print("année = ", key)
        print("n = ", n)
        print("moyenne historique [pnull] = ", pnull)
        print("moyenne de l'année [phat] = ", phat)
        #sm.stats.proportions_ztest(phat * n, n, pnull, alternative='larger')
        zstat, p_value = stm.stats.ztest(pDf[pColYName][pDf['Année']==key], value = pnull ,alternative='two-sided')        
        #zstat, p_value = stm.stats.proportions_ztest( nobs=n, pnull,  alternative='two-sided')
        print("zstat: ", zstat)
        print("p_value: ", p_value)
        print("h0: moyenne historique = moyenne [2014-2016]")
        print("h1: moyenne historique != moyenne [2014-2016]")
        if (p_value < pConf):
            print("p_value < confidence --> on REJÈTE l'hypothèse null ")
        else:
            print("p_value > confidence --> on accèpete l'hypothèse null")

# %%
print_analyse_hypothese('Transmittance produit (%)', moyenne_transmit_toutes_annees, dictAnneeTransmittance)
print_analyse_hypothese('Sucre sève (%)', moyenne_sucre_toutes_annees, dictAnneeSucre)

# %%
## https://towardsdatascience.com/demystifying-hypothesis-testing-with-simple-python-examples-4997ad3c5294
## Transmittance produit (%)
n =len(dfNumerique)
pnull = moyenne_transmit_toutes_annees ##0.43732148760330574 #moyenne hystorique
### Transmittance
confidence = 0.05
for key in dictAnneeTransmittance.keys():
    phat = dictAnneeTransmittance.get(key)
    print("année = ", key)
    print("n = ", n)
    print("pnull = ", pnull)
    print("phat = ", phat)
    #sm.stats.proportions_ztest(phat * n, n, pnull, alternative='larger')
    zstat, p_value = stm.stats.ztest(dfNumerique['Transmittance produit (%)'][dfNumerique['Année']==key], value = pnull ,alternative='two-sided')
    #zstat, p_value = stm.stats.proportions_ztest(44.41 * n, n, 44.036, alternative='larger')
    #zstat, p_value = stm.stats.proportions_ztest( nobs=n, pnull,  alternative='two-sided')
    print("zstat: ", zstat)
    print("p_value: ", p_value)
    if (p_value < confidence):
        print("p_value < confidence --> on rejète l'hypothèse null ")
    else:
        print("p_value > confidence --> on accèpete l'hypothèse null")
    


# %%
#dataframe[dataframe['Percentage'] > 70]
dfNumerique[dfNumerique['Année'] == 2014]
#dfNumerique[dfNumerique['année']>2014]

# %%
#import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
n = 1018
pnull = 52
phat = 56
stm.stats.proportions_ztest(phat * n, n, pnull, alternative='larger')

# %%


# %%

pColName = 'Année'
pListeExclusion = [2014, 2015, 2016]
pData = dfNumerique
pData[pColName][~pData[pColName].isin(pListeExclusion)].unique()

# %%


# %%
### moyenne transmittance

### boxplot vars y
#https://seaborn.pydata.org/tutorial/regression.html
i=0
for key in (dict_y_colsX.keys()):    
    print("key: ", key)
    #dfX = dict_y_colsX.get(key)
    #dfY = dfNumerique_Y_cols[key]    
    #i+=1
    #plt.figure(i)    
    #sns.histplot(data=dfY, bins=10)
    #plt.show(block=False)

# %%
dfNumerique['Année'].unique()

# %%
sns.scatterplot(data = dfNumerique
                , x = dfNumerique['Année']
                , y = dfNumerique['Transmittance produit (%)']
           )

# %%
sns.relplot(data = dfNumerique, x="Année", y="Transmittance produit (%)", hue="Année")

# %%
## boxenplot
def plotBoxenplot(pColNameY, pColNameX = "Année", pData = dfNumerique):
    i = random.randint(100)
    plt.figure(i)
    sns.boxenplot(data = pData, x=pColNameX, y=pColNameY)
    plt.show(block=False)

# %%
sns.boxenplot(data = dfNumerique, x="Année", y="Transmittance produit (%)")

# %%
sns.boxenplot(data = dfNumerique, x="Année", y="Sucre sève (%)")

# %%
### Q2
### plot kde graph
def plotKde(pColNameX, pHue='Année', pData = dfNumerique):
    plt.figure() 
    sns.kdeplot(data = pData, x=pColNameX, hue=pHue)
    plt.show(block=False)

# %%
###kdeplot
#sns.kdeplot(data = dfNumerique[dfNumerique['Année']>2012], x="Transmittance produit (%)", hue="Année")
#sns.kdeplot(data = dfNumerique, x="Transmittance produit (%)", hue="Année")
plotKde(pColNameX = "Transmittance produit (%)")

# %%
plotKde(pColNameX = "Sucre sève (%)")

# %%
"""
\------------\
En principe, un examen visuel ne montre pas une différence "significative"
ni de la transmittance ni du sucre
\--------------\
"""

# %%


# %%


# %%
dfNumerique.columns.values

# %%
"""
##### feature selection ####

https://machinelearningmastery.com/feature-selection-for-regression-data/

"""

# %%
# example of correlation feature selection for numerical data
# compare different numbers of features selected using mutual information
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

# %%

# define dataset
X = dfNumerique_X_cols.copy()

y = dfNumerique['Débit sève (L/j)']

# define the evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])

# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)]



# %%
### https://www.youtube.com/watch?v=VCVhwjbI6h8
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(dfNumerique_X_cols_vif, dfNumerique_Y_cols, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(dfNumerique_X_cols_vif, dfNumerique_Y_cols['Production moyenne par entaille (L)'], test_size=0.25, random_state=0)

# %%
## transforming data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# %%
## Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

regressor =  LinearRegression()
regressor.fit(X_train, y_train)


# %%
y_pred = regressor.predict(X_test)
math.sqrt(mean_squared_error(y_test, y_pred))
r2_score(y_test, y_pred)

# %%
for m in lst_models:
    print("==== SUMMARY ====")
    m.summary()


# %%
#dfNumerique_X_cols_vif.head()

# %%
duncan_prestige = stm.datasets.get_rdataset("Duncan", "carData")
Y = duncan_prestige.data['income']
X = duncan_prestige.data['education']
X = stm.add_constant(X)
print("type(Y)", type(Y))
print("type(X)", type(X))
print("len(Y): ", len(Y))
print("len(X): ", len(X))

# %%
#print(Y.shape)
X.head()
#Y.head()

# %%

model = stm.OLS(Y,X)
results = model.fit()
results.params

# %%
"""
###### 
"""

# %%
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

# %%
### pandas.series condition https://www.youtube.com/watch?v=BgfvF6mu20c
### pour chaque var Y, imprimer les variables les plus correlées
### on applique la condition 
for col in list_cols_dependennt_vars:
    print("===========================")
    print(" y = ", col.name)
    cond_correl_plus_grand_50 = ((corr[col.name] > 0.5) & (corr[col.name] <1)) 
    print(corr[col.name][cond_correl_plus_grand_50].sort_values(ascending=False))
    
#list_cols_dependennt_vars

# %%


# %%
sns.pairplot(dfNumerique)

# %%
"""
### At this point, we have strong relationships:
- Débit sève (L/j) --> Alimentation osmoseur (L/j) [0.97]
- Sucre sève (%) --> Pression osmoseur (bar) [0.9]
- Transmittance produit (%) --> Pression osmoseur (bar)[0.81], Sucre sortie osmoseur (%) [0.68]
- Production moyenne par entaille (L) --> Nombre épisodes gel/dégel [0.95]
"""

# %%
## Transmitance
### pression_osmoseur_vs_transmittance
sns.lmplot(x='Pression osmoseur (bar)', y = 'Transmittance produit (%)', data = dfNumerique)

# %%
## Le graphique ression_osmoseur_vs_transmittance montre une concentration
## dans x près de 40. Regardons s'il y a des outliers:
sns.boxplot(x='Pression osmoseur (bar)',data = dfNumerique)

# %%
## effectivement, le boxplot nous montre la présence des outliers
stats = dfNumerique['Pression osmoseur (bar)'].describe()
stats

# %%
sns.lmplot(x='Alimentation osmoseur (L/j)' , y='Débit sève (L/j)', data=dfNumerique)

# %%


# %%


# %%
### regardons combien de 0:
sns.histplot(x='Pression osmoseur (bar)',data = dfNumerique)

# %%
#sns.pairplot(dfNumerique)

# %%
## correlation seulement entre les variables indépendantes

dfDependantVars= dfNumerique.loc[:, ~dfNumerique.columns.isin([col.name for col in list_cols_dependennt_vars])]
type(dfDependantVars)

corr2 = dfDependantVars.corr()
sns.heatmap(corr2, cbar=True, annot=True, cmap="Blues", fmt=".02f", center=0)

# %%
"""
Multiple Linear Regression - https://www.youtube.com/watch?v=J_LnPL3Qg70

"""

# %%
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

# %%
print("coefs: ", list_reg[0].coef_.round(2))
print("intercept: ", list_reg[0].intercept_)

# %%
"""
fonction vérif distr normale
"""

# %%
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