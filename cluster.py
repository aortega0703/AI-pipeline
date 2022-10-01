import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import norm

def set_axis_labels(ax, x, y, z=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if z != None:
        ax.set_zlabel(z)


def match(U, Ud):
    D = norm.dist_matrix(U.T, Ud.T, norm.norm["Manhattan"])
    return U[np.argmin(D, axis=0), :]

def plot(X, U, Ud=None, C=None,
                 title="Clusters",
                 axes_names=["$X_0$", "$X_1$", "$Y_0$"],
                 cluster_names=None):
    C_N = U.shape[0]
    compare = Ud is not None
    error = 0

    cmap = plt.get_cmap("viridis")
    cnorm = mpl.colors.Normalize(vmin=0, vmax=C_N-1)
    plt.figure()
    ax = plt.axes(projection="3d", title=title)

    U = np.argmax(U, axis=0)
    Ud = np.argmax(Ud, axis=0) if compare else U
    if cluster_names is None:
        cluster_names = [f"C{c}" for c in range(C_N)]
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

    set_axis_labels(ax, *axes_names)
    if compare:
        ax.scatter([], [], [], color='k', marker="x", label=f"Errors: {error}")
    ax.legend()
