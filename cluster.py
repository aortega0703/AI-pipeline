import norm_space

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import itertools as iter


def axis_labels(ax, x, y, z=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if z != None:
        ax.set_zlabel(z)


def match(U, Ud):
    D = norm_space.dist_matrix(U.T, Ud.T,
        norm=norm_space.norm["Manhattan"])
    min_p = None
    min_d = np.inf
    for p in iter.permutations(range(U.shape[0])):
        d = np.sum([D[i, n] for n, i in enumerate(p)])
        if d < min_d:
            min_p, min_d = p, d
    return U[min_p, :]


def plot(ax, X, U, Ud=None, C=None,
                 title="Clusters",
                 axes_names=["$X_0$", "$X_1$", "$Y_0$"],
                 cluster_names=[]):
    C_N = U.shape[0]
    compare = Ud is not None
    error = 0

    cmap = plt.get_cmap("viridis")
    cnorm = mpl.colors.Normalize(vmin=0, vmax=C_N-1)
    ax.set_title(title)

    U = np.argmax(U, axis=0)
    Ud = np.argmax(Ud, axis=0) if compare else U

    for c in range(len(cluster_names), C_N):
        cluster_names.append(f"C{c}")

    for c in range(C_N):
        x = U == c
        xT = x & (Ud == c)
        xF = x & (Ud != c)
        c_name = cluster_names[c] + f": {np.sum(xT)}"
        if compare:
            e_c = np.sum(xF)
            c_name += f" - {e_c}"
            error += e_c
        ax.scatter(*X[:3, xT], color=cmap(cnorm(c)), label=c_name)
        ax.scatter(*X[:3, xF], color=cmap(cnorm(c)), marker="x")
        if C is not None:
            ax.scatter(*C[:3, c], color=cmap(cnorm(c)),
                       marker="*", edgecolors="black", s=40)

    axis_labels(ax, *axes_names[:3])
    if compare:
        ax.scatter([], [], [], color='k', marker="x", label=f"Errors: {error}")
    ax.legend()
