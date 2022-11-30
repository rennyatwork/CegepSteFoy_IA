import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_results(X, y, XX, YY, F, sF, axes):
    """
    Affichage des résultats de classification
    
    REMARQUE IMPORTANTE: le code ci-dessous a été écrit à l'arrache et n'est absolument pas un modèle de beau code...
    """
    sns.scatterplot(x=X[1,:], y=X[2,:], hue=y.ravel(), ax=axes[0], s=50)
    CS = axes[0].contour(XX,YY,F,[0], colors=["g"])
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")

    labels = ['Frontière de décision']
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    axes[0].legend(loc='upper left')

    CS2 = axes[1].contourf(XX, YY, sF, 100, cmap="plasma")
    CS = axes[1].contour(XX,YY,F,[0], colors=["k"])
    cbar = plt.colorbar(CS2) # Probability bar
    cbar.set_label('$P(y=1|x;theta)$')
    