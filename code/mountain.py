import norm_space

import numpy as np
import itertools as iter

def train(X, sigma, beta, epsilon=0, n=None, norm=norm_space.norm["Euclidean2"], substractive=False):
    def f(v, a): return np.exp(-norm(v)/(2 * a**2))
    if substractive or n is None:
        points = [v[:, None] for v in X.T]
    else:
        points = [np.array(v)[:, None] 
            for v in iter.product(*[np.linspace(0, 1, n)
            for _ in range(X.shape[0])])]
    m = [np.sum(f(X - v, sigma)) for v in points]
    mc, mc_new = 0, np.inf
    C = []
    while True:
        c = np.argmax(m)
        if m[c] <= epsilon:
            break
        C.append(points[c])
        m = [np.max(m[v] - m[c] * f(points[c] - points[v], beta), 0)
             for v in range(len(m))]
    return np.concatenate(C, axis=1)
