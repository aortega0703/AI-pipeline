import numpy as np

kernel = {
    "Linear" :
        lambda X, Y:
            X.T @ Y,
    "Polynomial" : 
        lambda d, c: lambda X, Y:
            (X.T @ Y + c)**d,
    "Radial":
        lambda gamma: lambda X, Y:
            np.squeeze(np.exp(-gamma *
                np.linalg.norm(X.T[:, None, :] - Y.T[None,:,:], axis=2)))
}

def train(X, Y, epochs, tolerance, eta=1, K=kernel["Linear"]):
    Y = np.argmax(Y, axis=0, keepdims=True).T*2-1
    dots = Y * K(X, X) * Y.T
    alpha = np.random.rand(X.shape[1], 1)
    cost = []
    for e in range(epochs):
        dalpha = np.ones((X.shape[1], 1)) - dots * alpha.T @ np.ones((X.shape[1], 1))
        cost.append(np.sum(alpha) - np.sum(dalpha))
        alpha += eta*dalpha
        if len(cost) > 2 and np.abs(cost[-2] - cost[-1]) < tolerance:
            print(e+1)
            break
        if (e+1) % (epochs*0.1) == 0:
            print(f"{e+1}/{epochs} ({(e+1)/epochs:.2%})  ", end="\r")
    W = np.sum(X * alpha.T * Y.T, axis=1, keepdims=True)
    return W, alpha, cost

def eval(X, alpha, X_train, Y_train, K=kernel["Linear"]):
    Y_train = np.argmax(Y_train, axis=0, keepdims=True).T*2-1
    out = np.sum(alpha * Y_train * K(X_train, X), axis=0, keepdims=True) <= 0
    return np.concatenate([out, ~out], axis=0)