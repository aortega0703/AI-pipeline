import numpy as np

regression = {
    "Linear": (
        lambda Y: Y,
        lambda Y: Y
    ),
    "Logistic": (
        lambda Y: np.log(Y/(1-Y)),
        lambda Y: 1/(1+np.exp(-Y)) - 0.5
    )
}

def train(X, Y, k = regression["Linear"]):
    X = np.concatenate([np.ones((1, X.shape[1])), X], axis=0)
    Y = k[0](Y*0.98 + 0.01)
    B = Y @ X.T @ np.linalg.inv(X @ X.T)
    return B


def eval(X, B, k = regression["Linear"]):
    X = np.concatenate([np.ones((1, X.shape[1])), X], axis=0)
    Y = B @ X
    U = (k[1](Y))[None, :]
    U = np.concatenate([U >= 0, U < 0], axis=0)
    return U