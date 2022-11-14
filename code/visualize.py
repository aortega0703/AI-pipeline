import numpy as np
import matplotlib.pyplot as plt

# Sets the name for axis x, y and z 
def axis_labels(ax, x, y, z=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if z != None:
        ax.set_zlabel(z)

# Plots a cloud of points in 3d with colours depending on their classification.
# If Ud is proportioned, wrongly classified data will be marked with an X and
# be accounted for in the legend. If C is used, it will plot those points
# as cluster centers in bold *. If fuzzy is True, then the plot is instead of
# 2 axis only and the third one is their degree of belonging.
def cluster(ax, X, U, Ud=None, C=None, fuzzy=False,
                 title="Clusters",
                 axes_names=["$X_0$", "$X_1$", "$X_2$"],
                 cluster_names=[]):
    C_N = U.shape[0]
    compare = Ud is not None
    F_N = 0

    ax.set_title(title)

    U_i = np.argmax(U, axis=0)
    Ud_i = np.argmax(Ud, axis=0) if compare else U_i

    for c in range(len(cluster_names), C_N):
        cluster_names.append(f"C{c}")

    if not fuzzy:
        U = U*0 + X[[2], :]
        if C is not None:
            C = C.copy()
            C[3, :] = 1

    for c in range(C_N):
        # Get hits and misses
        x = U_i == c
        xT = x & (Ud_i == c)
        xF = x & (Ud_i != c)
        xT_N = np.sum(xT)
        xF_N = np.sum(xF)
        x_N = np.sum((Ud_i == c))
        
        # Cluster label
        c_name = cluster_names[c] + f": {xT_N}"
        if compare:
            c_name += f"/{x_N} ({xT_N/x_N:.2%})"
            F_N += xF_N

        # Colors
        color = lambda N: ((c + 1)/C_N) * np.ones(N)
        ax.scatter(*X[:2, xT], U[c, xT], label=c_name,
            s=5, alpha=0.1, c=color(xT_N), cmap="jet", vmin=0, vmax=1)
        ax.scatter(*X[:2, xF], U[c, xF], marker="x",
            s=5, alpha=0.1, c=color(xF_N), cmap="jet", vmin=0, vmax=1)
        if C is not None:
            ax.scatter(*C[:3, c], marker="*", edgecolors="black", 
                s=40, alpha=1, color = "black")

    axis_labels(ax, *axes_names[:3])
    if compare:
        ax.scatter([], [], [], color='k', marker="x", label=f"Errors: {F_N}/{U.shape[1]} ({F_N/U.shape[1]:.2%})")
    legend = ax.legend(loc="upper right")
    for l in legend.legendHandles:
        l._sizes = [30]
        l._alpha = 1

def fig_size(y, x): return (x*4, y*4)

def plot(X, U_sets, suptitle):
    size = len(U_sets), len(list(U_sets.values())[0])
    fig, ax = plt.subplots(*size, figsize=fig_size(*size), layout="tight",
        subplot_kw={'projection': '3d'})
    fig.suptitle(suptitle)
    if type(ax) != list:
        ax = np.array([[ax]])
    for r, (name, U) in enumerate(U_sets.items()):
        for c, (u_name, u) in enumerate(U.items()):
            cluster(ax[r,c], X[name][0], u, title=u_name)
    return fig