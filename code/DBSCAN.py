import norm_space
import preprocess

import numpy as np

def run(X, P, epsilon, norm = norm_space.norm["Euclidean2"]):
    C = 0
    label = [None] * X.shape[1]
    for i in range(X.shape[1]):
        if label[i] != None:
            continue
        N = rangeQuery(X, i, epsilon, norm)
        if len(N) < P:
            label[i] = 0
            continue
        C += 1
        label[i] = C
        S = set()
        S_next = N - {i}
        visited = set()
        while len(S_next) != 0:
            S = S_next
            S_next = set()
            for j in S:
                visited.add(j)
                if label[j] == -1:
                    label[j] = C
                if label[j] != None:
                    continue
                label[j] = C
                N = rangeQuery(X, j, epsilon, norm)
                if len(N) >= P:
                    S_next = (S_next.union(N)) - visited
    return preprocess.vectorize(label)

def rangeQuery(X, i, epsilon, norm):
    distances = norm(X - X[:, [i]])
    neighbour = np.argwhere(distances <= epsilon).T[1]
    return set(neighbour.tolist())