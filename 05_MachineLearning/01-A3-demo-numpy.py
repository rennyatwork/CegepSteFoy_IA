#!/usr/bin/env python
# coding: utf-8

# **420-A52-SF - Algorithmes d'apprentissage supervisé - Automne 2022 - Spécialisation technique en Intelligence Artificielle**<br/>
# MIT License - Copyright (c) 2022 Mikaël Swawola
# <br/>
# ![Démonstration](static/01-A3-banner.png)

# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 0 - Chargement des bibliothèques

# In[3]:


# Manipulation de données
import numpy as np

# Visualisation de données
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# Configuration de la visualisation
sns.set(style="darkgrid", rc={'figure.figsize':(8.7,6.27)})


# ### 1 - Simulation d'un jeu de données

# In[34]:


# nb points
m = 1000


# In[35]:


np.random.seed(2020) # Pour la reproductibilité des résultats
theta_0 = 1
theta_1 = 2
x = np.linspace(1,m,m)
y = theta_0 + (theta_1 * x) + np.random.randint(-2,2,m)


# In[36]:


fig, ax = plt.subplots()
ax.scatter(x,y)
ax.set_xlabel("x")
ax.set_ylabel("y")


# In[37]:


def hypothesis(x, theta_0, theta_1):
    return theta_0 + theta_1 * x


# In[38]:


y_hat = hypothesis(x, theta_0, theta_1)


# In[39]:


print(f'x = {x}')
print(f'y_hat = {y_hat}')


# In[40]:


fig, ax = plt.subplots()
ax.scatter(x,y, label="Données")
ax.plot(x, theta_0 + (theta_1 * x), color="g", label='Modèle')
ax.scatter(x, y_hat, color="r", marker="+", s=92, linewidth=2, label="Prédictions")

for i,line in enumerate(x):
    ax.plot([x[i], x[i]] ,[y[i], y_hat[i]], linestyle=":", color = "r", label=f"Erreur {i}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc='upper left')


# ### 2 - Décomposition du calcul de l'erreur quadratique moyenne

# $J(\theta_{0},\theta_{1})= \frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}$

# In[41]:


print(f'y = {y}')
print(f'y_hat = {y_hat}')


# In[42]:


error = y_hat - y
error


# In[43]:


squared_error = (y_hat - y)**2
squared_error


# In[44]:


sse = np.sum((y_hat - y)**2)
sse


# In[45]:


mse = np.sum((y_hat - y)**2) / m
mse


# In[46]:


print(f'Erreur quadratique moyenne (version vectorisée) = {mse}')


# In[47]:


get_ipython().run_cell_magic('timeit', '', 'np.sqrt(np.sum((y_hat - y)**2) / m)\n')


# ### 3 - Calcul de l'erreur quadratique moyenne à l'aide d'une boucle for

# In[48]:


get_ipython().run_cell_magic('timeit', '', 'sse = 0\nfor i, _ in enumerate(x):\n    y_hat = hypothesis(x[i], theta_0, theta_1)\n    diff_squared = (y_hat - y[i])**2\n    sse = sse + diff_squared\nmse = sse / m\n')


# In[ ]:


print(f'Erreur quadratique moyenne (boucle for) = {mse}')


# In[ ]:


np.sqrt(mse)


# In[ ]:


# Calculer rapport des mesures de temps d'exécution

