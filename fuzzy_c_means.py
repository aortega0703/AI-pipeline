import norm_space

import numpy as np

def update_centers(X, U, m):
    C = np.zeros((X.shape[0], U.shape[0]))
    Um = U**m
    for c in range(C.shape[1]):
        C[:, c] = np.sum(Um[c, :] * X[:, :], axis=1) / np.sum(Um[c, :])
    return C


def eval(X, C, m, norm=norm_space.norm["Euclidean2"]):
    D = norm_space.dist_matrix(C, X, norm)**(1/(m-1))
    U = np.zeros((C.shape[1], X.shape[1]))
    for x in range(X.shape[1]):
        U[:, x] = 1/(D[:, x] * np.sum(1/D[:, x]))
    U[D == 0] = 1
    return U


def run(c, X, m, epsilon, cost=None):
    if cost is None:
        def cost(X, U, C, m, norm=norm_space.norm["Euclidean2"]):
            D = norm_space.dist_matrix(C, X, norm)
            return np.sum((U[:, :]**m) * D[:, :])
    U = np.random.random((c, X.shape[1]))
    U /= np.sum(U, axis=0)
    C = update_centers(X, U, m)
    J, J_new = np.inf, 0
    while np.abs(J - J_new) > epsilon:
        J = J_new
        C = update_centers(X, U, m)
        U = eval(X, C, m)
        J_new = cost(X, U, C, m)
    return U, C
