import norm_space
import numpy as np
from scipy.special import expit

sigmoid = (
    lambda x: expit(x),
    lambda x: sigmoid[0](x) * (1 - sigmoid[0](x)))
softmax = (
    lambda x: np.exp(x) / np.sum(np.exp(x), axis=0),
    lambda X: softmax[0](X).T[:,:,None] *
        (np.eye(X.shape[0]) - softmax[0](X).T[:,None,:]))


def feedforward(X, W, B, phi, Yd):
    k = len(W) - 1
    V = [None] * (k+1)
    Y = [None] * (k+1)
    Y[0] = X

    for l in range(1, k+1):
        V[l] = W[l] @ Y[l-1]
        if B != None:
            V[l] += B[l]
        Y[l] = phi[l][0](V[l])
    
    E = np.average(norm_space.norm["Manhattan"](Y[k] - Yd))/2
    dE = Y[k] - Yd
    return Y, V, E, dE

def backpropagate(W, V, dE, phi):
    p = V[-1].shape[1]
    k = len(W) - 1
    delta = [None] * (k+1)

    delta[k] = phi[-1][1](V[k])
    
    if len(delta[k].shape) != 2:
        temp = (delta[k] @ dE.T[:,:,None])
        delta[k] = np.squeeze(temp).T
    else:
        delta[k] = delta[k] * dE

    for l in reversed(range(1, k)):
        delta[l] = (W[l+1].T @ delta[l+1]) * phi[l][1](V[l])
    return delta


def update(X, Yd, W, B, eta, phi):
    p = X.shape[1]
    k = len(W) - 1

    # feedforward
    Y, V, E, dE = feedforward(X, W, B, phi, Yd)
    # backpropagation
    delta = backpropagate(W, V, dE, phi)

    # update weights
    for l in range(1, k+1):
        W[l] -= eta * (delta[l] @ Y[l-1].T) / p
        if B != None:
            B[l] -= eta * (delta[l] @ np.ones((p, 1))) / p
    return W, B, delta

def eval(X, W, B=None, phi=sigmoid, classify=False, Yd=0):
    if type(phi) != list:
        phi = [phi] * len(W)
    if classify:
        phi[-1] = softmax

    Y, _, E, _ = feedforward(X, W, B, phi, Yd)
    return Y[-1], E

def train(train_set, epochs, hidden, eta, phi=sigmoid,
          classify=False, test_set={}, W=None, B=None):
    neurons = [train_set[0].shape[0], *hidden, train_set[1].shape[0]]
    k = len(neurons) - 1
    test_set["Train"] = train_set

    if W is None:
        W = [None] * (k+1)
        for l in range(1, k+1):
            W[l] = np.random.randn(neurons[l], neurons[l-1])
    if B is None:
        B = [None] * (k+1)
        for l in range(1, k+1):
            B[l] = np.random.randn(neurons[l])[:, None]
    
    if type(phi) != list:
        phi = [phi] * len(neurons)
    if classify:
        phi[-1] = softmax

    delta = []
    E = {k: [] for k in test_set.keys()}

    for e in range(epochs):
        W, B, delta_curr = update(train_set[0], train_set[1], W, B,
            eta, phi)
        delta_curr = [np.mean(norm_space.norm["Euclidean2"](delta_curr[l]))
            for l in range(1, k+1)]
        delta.append(delta_curr)

        for t_name, t_set in test_set.items():
            if type(t_set) == tuple:
                t_X, t_Yd = t_set
            else:
                t_X, t_Yd = t_set, None
            _, t_E = eval(t_X, W, B, phi, classify, t_Yd)
            E[t_name].append(t_E)
        
        if (e+1) % (epochs*0.1) == 0:
           print(f"{e+1}/{epochs} ({e/epochs:.0%}):")
           print(f"\tMean Gradient Length: {np.mean(delta_curr)}")
    return W, B, delta, E