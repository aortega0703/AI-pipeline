import numpy as np


# lower bound to sample size
def PAC_eta(H_norm, delta, epsilon):
    return (np.log(H_norm) - np.log(delta))/epsilon

# lower bound to generalization error
def PAC_delta(H_norm, epsilon, eta):
    return H_norm/np.exp(eta*epsilon)

# TN FP
# FN TP
def confusion(U1, U2, supervise = True):
    U1 = np.argmax(U1, axis = 0)
    if supervise:
        Ud = np.argmax(Ud, axis = 0)
        CM = np.zeros((U1.shape[0], U1.shape[0]))
        for u in range(len(U1)):
            CM[Ud[u], U1[u]] += 1
        return CM
    else:
        # Compares if pairs of points share cluster under U1 and U2 or not
        U2 = np.argmax(U2, axis=0)
        CM = np.zeros((2,2))
        for u in range(len(U1)):
            for v in range(u):
                p1 = U1[u] == U1[v]
                p2 = U2[u] == U2[v]
                CM += [[not p1 and not p2, p1 and not p2], [not p1 and p2, p1 and p2]]
        return CM


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

