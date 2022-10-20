import numpy as np

def train(X, Y, I):
    if np.sum(Y[0, I]) == 0:
        return [False]
    if np.sum(I) == np.sum(Y[0, I]):
        return [True]
    def H(P):
        return -np.sum(P*np.log2(P), axis=1)
    N = np.sum(I)
    split = np.mean(X[:, I], axis=1,keepdims=True)
    left = I & (X <= split)
    right = I & (X > split)
    p1 = np.array([np.sum(Y[0, row]) for row in left])
    np1 = np.sum(left, axis=1) - p1
    p2 = np.array([np.sum(Y[0, row]) for row in right])
    np2 = np.sum(right, axis=1) - p2
    L = H(np.array([p1/N, np1/N, p2/N, np2/N]).T)
    c = np.squeeze(np.argmax(L))
    return [(c, float(split[c])), train(X, Y, left[c, :]), train(X, Y, right[c, :])] 

def eval(X, T, I):
    if len(T) == 1:
        if T[0]:
            return I
        else:
            return I == np.inf 
    root = T[0]
    left = I & (X[root[0], :] <= root[1])
    right = I & (X[root[0], :] > root[1])
    return eval(X, T[1], left) | eval(X, T[2], right)