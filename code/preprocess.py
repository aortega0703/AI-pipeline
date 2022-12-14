import pandas as pd
import numpy as np

# Reads the data from a csv file and drops all nan values. if header or index
# are provided, they are dropped
def get_data(file, header = None, index_col=None):
    data = pd.read_csv(file, index_col=index_col, header=header)
    data = data.dropna().reset_index(drop=True)
    I_S = data.index
    N_S = len(I_S)
    return data, N_S, I_S

# Given a matrix "data", split it by column "out_col"
def split_XY(data, out_col):
    X = data.iloc[:, :out_col]
    X_name = X.columns.to_numpy()
    Y = data.iloc[:, out_col:]
    Y_name = Y.columns.to_numpy()
    return X.to_numpy(), Y.to_numpy(), X_name, Y_name

# Normalizes all data in "X_S" and "Y_S". If "Y_S" is categorical its contents
# are split into as many binary columns as categories present. The return
# values contain information to later reverse this step and collect the
# original data.
def preprocess(X_S, Y_S, classify):
    X_S, X_revert = normalize(X_S)
    
    if classify:
        Y_S, Y_revert = name_to_int(Y_S)
        Y_S = vectorize(Y_S[0,:])
    else:
        Y_S, Y_revert = normalize(Y_S)
    return X_S, Y_S, X_revert, Y_revert

# Translates and scales the coordenates of each axis so that they fall in the
# hypercube (0,1)^d
def normalize(points):
    min = np.min(points, axis=1, keepdims=True)
    scale = np.max(points, axis=1, keepdims=True) - min
    def denormalize(points): return points * scale + min
    return (points - min) / scale, denormalize

# Receives categorical data and assigns a number to each category to be replaced to.
def name_to_int(data):
    names = np.unique(data)
    name_map = dict(map(reversed, enumerate(names)))
    data = np.array([[name_map[x] for x in data[0, :]]])
    if len(names) == 2:
        names = ["False", "True"]
    return data, names

# Given an integer column, replaces each entry with a binary column with the only
# 1 value being on the appropiate index.
def vectorize(data):
    I = np.identity(np.max(data) + 1)
    return I[:, data]
