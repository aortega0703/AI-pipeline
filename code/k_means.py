import norm_space

import numpy as np

def update_centers(X, U):
    C = np.zeros((X.shape[0], U.shape[0]))
    for c, u in enumerate(U):
        C[:, c] = np.average(X[:, u == 1], axis=1)
    return C

def eval(X, C, norm=norm_space.norm["Euclidean2"]):
    U = np.zeros((C.shape[1], X.shape[1]), dtype=int)
    D = norm_space.dist_matrix(C, X, norm)
    I = np.identity(C.shape[1])
    for x, c in enumerate(np.argmin(D, axis=0)):
        U[:, x] = I[:, c]
    return U


def train(k, X, epsilon, norm=norm_space.norm["Euclidean2"], cost=None):
    if cost is None:
        def cost(X, U, C, norm=norm_space.norm["Euclidean2"]):
            return np.sum(norm(X - C[:, np.argmax(U, axis=0)]))
    C = X[:, np.random.randint(0, X.shape[1], size=k)]
    J, J_new = np.inf, 0
    while np.abs(J - J_new) > epsilon:
        J = J_new
        U = eval(X, C)
        C = update_centers(X, U)
        J_new = cost(X, U, C, norm)
    return eval(X, C), C
