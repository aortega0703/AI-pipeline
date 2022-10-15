import numpy as np

def axis_labels(ax, x, y, z=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if z != None:
        ax.set_zlabel(z)

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
        x_N = xT_N + xF_N
        
        # Cluster label
        c_name = cluster_names[c] + f": {xT_N}"
        if compare:
            c_name += f"/{x_N} ({xT_N/x_N:.2%})"
            F_N += xF_N

        # Colors
        color = lambda N: (c/C_N) * np.ones(N)
        ax.scatter(*X[:2, xT], U[c, xT], label=c_name,
            s=5, alpha=0.1, c=color(xT_N), cmap="gnuplot", vmin=0, vmax=1)
        ax.scatter(*X[:2, xF], U[c, xF], marker="x",
            s=5, alpha=0.1, c=color(xF_N), cmap="gnuplot", vmin=0, vmax=1)
        if C is not None:
            ax.scatter(*C[:3, c], marker="*", edgecolors="black", 
                s=40, alpha=1, color = "black", vmin=0, vmax=1)

    axis_labels(ax, *axes_names[:3])
    if compare:
        ax.scatter([], [], [], color='k', marker="x", label=f"Errors: {F_N}/{U.shape[1]} ({F_N/U.shape[1]:.2%})")
    ax.legend(loc="upper right")