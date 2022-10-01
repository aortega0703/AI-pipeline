import norm
import numpy as np
from scipy.special import expit

def sigmoid(x): return expit(x)
def dsigmoid(x): return sigmoid(x) * (1 - sigmoid(x))
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=0)
def dsoftmax(x): return softmax(x) * (np.eye(x.shape[0]) - softmax(x).T)


def feedforward(X, W, B=None, phi=None, dphi=None, cluster=False, Yd=None, full=False):
    if phi is None:
        phi = sigmoid
    if dphi is None:
        dphi = dsigmoid
    k = len(W) - 1
    V = [None] * (k+1)
    Y = [None] * (k+1)
    Y[0] = X
    for l in range(1, k+1):
        V[l] = W[l] @ Y[l-1]
        if B != None:
            V[l] += B[l]
        Y[l] = phi(V[l])
    if cluster:
        Y[k] = softmax(V[k])
    if Yd is None:
        E = 0
    else:
        dE = Y[k] - Yd
        E = np.average(norm.norm["Euclidean2"](Y[k] - Yd))/2
    if full:
        return Y, V, E, dE
    else:
        return Y[k], E


def update(X, Yd, phi, dphi, W, B=None, cluster=False, eta=1):
    p = X.shape[1]
    k = len(W) - 1

    # feedforward
    Y, V, E, dE = feedforward(X, W, B,phi, dphi,  cluster, Yd, full=True)
    # backpropagation
    delta = [None] * (k+1)
    if cluster:
        delta[k] = np.concatenate([dsoftmax(V[k][:, [c]]) @ dE[:, [c]]
                                   for c in range(p)], axis=1)
    else:
        delta[k] = dphi(V[k]) * dE

    for l in reversed(range(1, k)):
        delta[l] = (W[l+1].T @ delta[l+1]) * dphi(V[l])

    # update
    for l in range(1, k+1):
        W[l] -= eta * (delta[l] @ Y[l-1].T) / p
        if B != None:
            B[l] -= eta * (delta[l] @ np.ones((p, 1))) / p
    return delta, E, W, B


def train(sets, name, hidden, epochs, phi=None, dphi=None,
          eta=1, bias=True, cluster=False):
    if phi is None:
        phi = sigmoid
    if dphi is None:
        dphi = dsigmoid
    neurons = [sets[name][0].shape[0], *hidden, sets[name][1].shape[0]]
    k = len(neurons) - 1

    W = [None] * (k+1)
    for l in range(1, k+1):
        W[l] = np.random.randn(neurons[l], neurons[l-1])
    if bias:
        B = [None] * (k+1)
        for l in range(1, k+1):
            B[l] = np.random.randn(neurons[l])[:, None]
    else:
        B = None

    for e in range(epochs):
        _, _, W, B = update(sets[name][0], sets[name][1], 
            phi, dphi, W, B, cluster, eta)
    return W, B