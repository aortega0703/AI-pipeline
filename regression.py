import norm_space

import numpy as np
from numpy.linalg import inv


def linear(X, Y):
    X = np.concatenate([np.ones((X.shape[1], 1)), X.T], axis=1)
    Y = Y.T
    B = inv(X.T @ X) @ X.T @ Y
    E = norm_space.norm["Euclidean2"](Y - (X @ B))
    return B, E


def linear_eval(X, B):
    X = np.concatenate([np.ones((X.shape[1], 1)), X.T], axis=1)
    return (X @ B).T


def logistic(X, P):
    P = (P.T * 0.98) + 0.01
    Y = np.log(P/(1-P))
    B, E = linear(X, Y.T)
    X = np.concatenate([np.ones((X.shape[1], 1)), X.T], axis=1)
    Y = X @ B
    Yd = 1/(1+np.exp(-Y))
    E = norm_space.norm["Euclidean2"](P - Yd)
    return B, E


def logistic_eval(X, B):
    X = np.concatenate([np.ones((X.shape[1], 1)), X.T], axis=1)
    Y = (X @ B)
    return (1/(1+np.exp(-Y))).T
