import numpy as np


# lower bound to sample size
def PAC_eta(H_norm, delta, epsilon):
    return (np.log(H_norm) - np.log(delta))/epsilon

# lower bound to generalization error
def PAC_delta(H_norm, epsilon, eta):
    return H_norm/np.exp(eta*epsilon)

# TN FP
# FN TP
def confusion(U1, U2, supervise = True, compact=True):
    p = U1.shape[1]
    if supervise:
        U1 = np.argmax(U1, axis = 0)
        U2 = np.argmax(U2, axis = 0)
        CM = np.zeros((np.max(U1)+1, np.max(U1)+1))
        for u in range(p):
            CM[U2[u], U1[u]] += 1
        return CM/p
    else:
        # Compares if pairs of points share cluster under U1 and U2 or not
        U1 = np.argmax(U1, axis = 0, keepdims=True) != 0
        U2 = np.argmax(U2, axis=0, keepdims=True) != 0
        equal_pw = lambda A: ~(A.T @ ~A) * ~(~A.T @ A)
        D1 = equal_pw(U1)
        D2 = equal_pw(U2)
        CM_T = np.array([
            [np.sum(~D1 * ~D2), np.sum(D1 * ~D2)],
            [np.sum(~D1 * D2), np.sum(D1 * D2) - U1.shape[1]]])
        return CM_T / 2



index = {
    "P": lambda CM: #Precision
        CM[0,0] / (CM[0,0] + CM[1,0]),
    "R": lambda CM: #Recall
        CM[0, 0] / (CM[0, 0] + CM[0, 1]),
    "F": lambda alpha: lambda CM: #F-alpha
        (1 + alpha) * index["P"](CM) * index["R"](CM) /
        (alpha * index["P"](CM) + index["R"](CM)),
    "Mu": lambda CM: #Mean
        (CM[0,0] + CM[0,1]) / np.sum(CM),
    "Sigma2": lambda CM: #Variance
        index["Mu"](CM) - index["Mu"](CM)**2,
    "Phi": lambda CM: #Phi
        (CM[0,0] * CM[1,1] - CM[0,1] * CM[1,0]) /
        ((CM[0, 0] + CM[0, 1]) * (CM[0, 0] + CM[1, 0])
         * (CM[0, 1] + CM[1, 1]) * (CM[1, 0] + CM[1, 1])),
    "Rand": lambda CM: #Rand
        (CM[0,0] + CM[1,1])/np.sum(CM)
}

