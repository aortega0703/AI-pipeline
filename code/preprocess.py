import pandas as pd
import numpy as np

def get_data(file, header = None, index_col=None):
    data = pd.read_csv(file, index_col=index_col, header=header).reset_index(drop=True)
    N_S = len(data.index)
    axes_names = data.columns.to_list()
    indices = data.index
    return data, N_S, axes_names, indices


def normalize(points):
    min = points.min(axis=1, keepdims=True)
    scale = points.max(axis=1, keepdims=True) - min
    def denormalize(points): return points * scale + min
    return (points - min) / scale, denormalize


def name_to_int(data):
    names = data.unique()
    name_map = dict(map(reversed, enumerate(names)))
    return data.apply(lambda x: name_map[x]), names


def vectorize(data):
    I = np.identity(np.max(data) + 1)
    return I[:, data]

def preprocess(data, cluster):
    X_S, Y_S = data.iloc[:, :-1], data.iloc[:, -1]
    X_S, denormalize_X = normalize(X_S.to_numpy().T)

    if cluster:
        Y_S, revert_Y = name_to_int(Y_S)
        Y_S = vectorize(Y_S.to_list())
    else:
        Y_S, revert_Y = normalize(Y_S.to_numpy()[None, :])
    return X_S, Y_S, denormalize_X, revert_Y
