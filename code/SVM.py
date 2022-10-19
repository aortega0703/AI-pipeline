import numpy as np

def SVM(X, U, epochs, tolerance, eta=1):
    Y = np.argmax(U, axis=0, keepdims=True).T*2-1
    dots = (Y * X.T) @ (X * Y.T)
    alpha = np.random.rand(X.shape[1], 1)
    b = np.random.rand()
    alpha_norm = []
    cost = []
    for e in range(epochs):
        W = np.sum(X * alpha.T * Y.T, axis=1)
        dalpha = np.ones((X.shape[1], 1)) - dots * alpha.T @ np.ones((X.shape[1], 1))
        cost.append(np.sum(alpha) - np.sum(dalpha))
        alpha_norm.append(np.squeeze(alpha.T @ alpha))
        alpha += eta*dalpha
        b += Y.T @ alpha
        if len(alpha_norm) > 2 and np.abs(alpha_norm[-2] - alpha_norm[-1]) < tolerance:
            print(e+1)
            break
        if (e+1) % (epochs*0.1) == 0:
            print(e+1)
    W = np.sum(X * alpha.T * Y.T, axis=1)
    return W, b, cost, alpha_norm