import norm_space

import numpy as np
from numpy.linalg import inv


def linear(X, Y):
    X = np.concatenate([np.ones((1, X.shape[1])), X], axis=0)
    B = Y @ X.T @ inv(X @ X.T)
    E = norm_space.norm["Euclidean2"](Y - (B @ X))
    return B, np.sum(E)


def linear_eval(X, B):
    X = np.concatenate([np.ones((1, X.shape[1])), X], axis=0)
    return B @ X


def logistic(X, P):
    P = (P * 0.98) + 0.01
    Y = np.log(P/(1-P))
    B, E = linear(X, Y)
    X = np.concatenate([np.ones((1, X.shape[1])), X], axis=0)
    Y_out = B @ X
    P_out = 1/(1+np.exp(-Y_out))
    E = norm_space.norm["Euclidean2"](P - P_out)
    return B, np.sum(E)


def logistic_eval(X, B):
    X = np.concatenate([np.ones((1, X.shape[1])), X], axis=0)
    Y = B @ X
    return 1/(1+np.exp(-Y))
