import numpy as np

kernel = {
    "Identity" :
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

def train(X, U, epochs, tolerance, eta=1, K=kernel["Identity"]):
    Y = np.argmax(U, axis=0, keepdims=True).T*2-1
    dots = Y * K(X, X) * Y.T
    alpha = np.random.rand(X.shape[1], 1)
    alpha_norm = []
    cost = []
    for e in range(epochs):
        W = np.sum(X * alpha.T * Y.T, axis=1)
        dalpha = np.ones((X.shape[1], 1)) - dots * alpha.T @ np.ones((X.shape[1], 1))
        cost.append(np.sum(alpha) - np.sum(dalpha))
        alpha_norm.append(np.squeeze(alpha.T @ alpha))
        alpha += eta*dalpha
        if len(alpha_norm) > 2 and np.abs(alpha_norm[-2] - alpha_norm[-1]) < tolerance:
            print(e+1)
            break
        if (e+1) % (epochs*0.1) == 0:
            print(e+1)
    W = np.sum(X * alpha.T * Y.T, axis=1, keepdims=True)
    return W, alpha, cost, alpha_norm

def eval(U, alpha, X, Y, Yd=None, K=kernel["Identity"]):
    Y = np.argmax(Y, axis=0, keepdims=True).T*2-1
    print(U.shape, Y.shape)
    out = np.sum(alpha * Y * K(X, U), axis=0, keepdims=True) <= 0
    print(out.shape)
    return np.concatenate([out, ~out], axis=0)