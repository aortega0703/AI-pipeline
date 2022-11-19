import norm_space

import numpy as np


# lower bound to sample size
def PAC_eta(H_norm, delta, epsilon):
    return (np.log(H_norm) - np.log(delta))/epsilon

# lower bound to generalization error
def PAC_delta(H_norm, epsilon, eta):
    return H_norm/np.exp(eta*epsilon)

# TN FP
# FN TP
# Calculates the confusion matrix given 2 belonging matrices. If supervise is
# False then it is calculated based on pairs of points rather than single
# points. If compact is False, then a confussion matrix is created with as many
# rows and columns as classes present.
def confusion(U1, U2, supervise = True, compact=True):
    p = U1.shape[1]
    A = np.argmax(U1, axis = 0, keepdims=True) != 0
    B = np.argmax(U2, axis=0, keepdims=True) != 0
    if not supervise:
        equal_pw = lambda M: ~(M.T @ ~M) * ~(~M.T @ M)
        A = equal_pw(A)
        B = equal_pw(B)
    CM = np.array([
            [np.sum(~A * ~B), np.sum(A * ~B)],
            [np.sum(~A * B), np.sum(A * B)]])
    if not supervise:
        CM[1, 1] -= p
        CM = CM/2
    return CM

# Definition of multiple indices
index = {
    "Sensitivity": lambda TN, FP, FN, TP: #Sensitivity
        TP/(TP + FN),
    "Specificity": lambda TN, FP, FN, TP: #Specificity
        TN/(TN + FP),
    "Precision": lambda TN, FP, FN, TP: #Precision
        TP/(TP + FP),
    "Accuracy": lambda TN, FP, FN, TP: #Accuracy
        (TP + TN) / (TN+FP+FN+TP),
    "Phi": lambda TN, FP, FN, TP: #Phi
        ((TP*TN) - (FP*FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
}

# Given a matix of belonging U_l and a desired classification Y, calculates all
# indices defined in "index" and returns a matrix with the results
def eval(U_l, Y):
    I = np.empty((len(index)+1, len(U_l)+1), dtype=object)
    I[1:, 0] = list(index.keys())
    for U_i, (U_name, U) in enumerate(U_l.items()):
        CM = confusion(U, Y, supervise=False).flatten()
        I[0, U_i+1] = U_name
        I[1:, U_i+1] = [i(*CM) for i in index.values()]
    return I

# Given a set of points X with their respective classification U, evaluates the
# Davies–Bouldin index
def DB(X, U, norm = norm_space.norm["Euclidean2"]):
    n = U.shape[0]
    U = np.argmax(U, axis=0)
    C = np.array([np.mean(X[:, U==i], axis=1) for i in range(n)]).T
    sigma = [np.mean(norm(X[:, U==i] - C[:, [i]])) for i in range(n)]

    acum = 0
    for i in range(n):
        curr_max = -np.inf
        for j in range(n):
            if i == j:
                continue
            curr = (sigma[i] + sigma[j]) / norm(C[:,i] - C[:,j])[0]
            curr_max = np.max([curr_max, curr])
        acum += curr_max
    return acum/n

def dunn(X, U, norm = norm_space.norm["Euclidean2"]):
    n = U.shape[0]
    U = np.argmax(U, axis = 0)
    C = np.array([np.mean(X[:, U==i], axis=1) for i in range(n)]).T

    num = np.inf
    den = -np.inf
    for i in range(n):
        den = np.max([den, 
            np.max(norm_space.dist_matrix(X[:, U==i], X[:, U==i]))])
        for j in range(i+1, n):
            num = np.min([num,
                norm(C[:, [i]] - C[:, [j]])[0]])
    return num/den