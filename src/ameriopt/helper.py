import numpy as np


#########################

# Legendre Polynomial

def laguerre_polynomials_ind(S, k):

    u0 = np.ones(S.shape)
    x1 = 1 - S
    x2 = 1 - 2*S + S**2/2
    x3 = 1 - 3*S + 3*S**2/2 - S**3/6
    x4 = 1 - 4*S + 3*S**2 - 2*S**3/3 + S**4/24

    X  = [np.stack([u0, x1, x2]),
          np.stack([u0, x1, x2, x3]),
          np.stack([u0, x1, x2, x3, x4])]

    return X[k-2]

def laguerre_polynomials(S, k):

    u0 = np.ones(S.shape)
    x1 = 1 - S
    x2 = 1 - 2*S + S**2/2
    x3 = 1 - 3*S + 3*S**2/2 - S**3/6
    x4 = 1 - 4*S + 3*S**2 - 2*S**3/3 + S**4/24

    X  = [np.stack([u0, x1, x2], axis = 1),
          np.stack([u0, x1, x2, x3], axis = 1),
          np.stack([u0, x1, x2, x3, x4], axis = 1)]

    return X[k-2]