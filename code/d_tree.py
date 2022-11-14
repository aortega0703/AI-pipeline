import numpy as np

# Creates a decision tree based on some given data
def train(X, Y, I):
    if np.sum(Y[0, I]) == 0:
        return [False]
    if np.sum(I) == np.sum(Y[0, I]):
        return [True]
    def H(P):
        return -np.sum(P*np.log2(P), axis=1)
    N = np.sum(I)
    # Splits every axis by is mean value
    split = np.mean(X[:, I], axis=1,keepdims=True)
    left = I & (X <= split)
    right = I & (X > split)
    p1 = np.array([np.sum(Y[0, row]) for row in left])
    np1 = np.sum(left, axis=1) - p1
    p2 = np.array([np.sum(Y[0, row]) for row in right])
    np2 = np.sum(right, axis=1) - p2
    # Chooses the split of greater entropy
    L = H(np.array([p1/N, np1/N, p2/N, np2/N]).T * 0.98 + 0.1)
    c = np.squeeze(np.argmax(L))
    return [(c, float(split[c])), train(X, Y, left[c, :]), train(X, Y, right[c, :])] 

# Evaluates the input on the current node of the Tree and returns the final
# class prediction recursively
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

# Calculates the dept of the given decision tree
def depth(T):
    if type(T) != list or len(T) == 1:
        return 1
    left = depth(T[1])
    right = depth(T[2])
    return np.max([left, right]) + 1