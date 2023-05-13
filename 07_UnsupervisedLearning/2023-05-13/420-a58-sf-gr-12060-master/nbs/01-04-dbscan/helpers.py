import numpy as np
from sklearn import datasets

def generate_toy_dataset(seed=2021):
    """
    Génère un jeu de données sour forme de smiley, pour illustration de l'algorithme DBSCAN
    """
    n_samples_circles = 2000
    n_samples_moons1 = 750
    n_samples_moons2 = 200
    n_samples_noise = 50

    circle = datasets.make_circles(n_samples=n_samples_circles,
                             noise = 0.02,
                             factor=0.8,
                             random_state=seed)

    moon = datasets.make_moons(n_samples=n_samples_moons1,
                             noise = 0.04,
                             random_state=seed)

    moon2 = datasets.make_moons(n_samples=n_samples_moons2,
                             noise = 0.1,
                             random_state=seed)

    moon3 = datasets.make_moons(n_samples=n_samples_moons2,
                             noise = 0.1,
                             random_state=seed+1)

    np.random.seed(seed)
    X_noise = np.random.uniform(-0.8,0.8,(n_samples_noise,2))

    X_circle = circle[0][circle[1] == 1,:]

    X_moon = moon[0][moon[1] == 1,:]
    X_moon[:,0] = 0.5*(X_moon[:,0] - 1)
    X_moon[:,1] = 0.4*(X_moon[:,1] - 0.8)

    X_moon2 = moon2[0][moon2[1] == 1,:]
    X_moon2 = (-0.1 * X_moon2)
    X_moon2[:,0] = X_moon2[:,0] - 0.3
    X_moon2[:,1] = X_moon2[:,1] + 0.3

    X_moon3 = moon3[0][moon3[1] == 1,:]
    X_moon3 = (-0.1 * X_moon3)
    X_moon3[:,0] = X_moon3[:,0] + 0.5
    X_moon3[:,1] = X_moon3[:,1] + 0.3

    X = np.concatenate((X_circle, X_moon, X_moon2, X_moon3, X_noise), axis=0)
    
    return X
