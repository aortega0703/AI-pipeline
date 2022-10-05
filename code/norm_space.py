import numpy as np
import itertools as iter
from numpy.linalg import inv

norm = {
    "Euclidean2": lambda x: np.sum(x**2, axis=0, keepdims=True),
    "Manhattan": lambda x: np.sum(np.abs(x), axis=0, keepdims=True),
    "Infinity": lambda x: np.max(np.abs(x), axis=0, keepdims=True),
    "Mahalanobis2": lambda S: lambda x: np.concatenate(
        [c[None, :] @ inv(S) @ c[:, None] for c in x.T])
}


def dist_matrix(X, Y, norm=norm["Euclidean2"]):
    D = np.zeros((X.shape[1], Y.shape[1]))
    for x, y in iter.product(range(X.shape[1]), range(Y.shape[1])):
        D[x, y] = norm(X[:, x] - Y[:, y])
    return D
