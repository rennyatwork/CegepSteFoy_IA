import numpy as np

def polynomial(X, degree=2):
    vec = [np.ones((X.shape[0],1))]
    for d in range(1,degree+1):
        temp = np.array((X**d))
        vec.append(temp)
    return np.hstack(vec)
