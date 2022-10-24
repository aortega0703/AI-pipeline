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
    A = np.argmax(U1, axis = 0, keepdims=True) != 0
    B = np.argmax(U2, axis=0, keepdims=True) != 0
    if not supervise:
        equal_pw = lambda A: ~(A.T @ ~A) * ~(~A.T @ A)
        A = equal_pw(U1)
        B = equal_pw(U2)
    CM = np.array([
            [np.sum(~A * ~B), np.sum(A * ~B)],
            [np.sum(~A * B), np.sum(A * B)]])
    if not supervise:
        CM[1, 1] -= p
        CM /= 2
    return CM


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

def eval(U_l, Y):
    I = np.empty((len(index)+1, len(U_l)+1), dtype=object)
    I[1:, 0] = list(index.keys())
    for U_i, (U_name, U) in enumerate(U_l.items()):
        CM = confusion(U, Y).flatten()
        I[0, U_i+1] = U_name
        I[1:, U_i+1] = [i(*CM) for i in index.values()]
    return I