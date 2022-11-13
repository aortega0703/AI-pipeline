# Module handling 3d plots of point clouds with assigned classes

# Own modules
import norm_space

# Pip modules
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import itertools as iter

# Sets x,y and z axes names for a 2d or 3d axis
def axis_labels(ax, x, y, z=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if z != None:
        ax.set_zlabel(z)

# Accepts 2 different classifications as input. Outputs the permutation of
# class naming that minimizes the error obtained with a manhattan norm
def match(U, Ud):
    # Distance matrix of classes vs classes
    D = norm_space.dist_matrix(U.T, Ud.T,
        norm=norm_space.norm["Manhattan"])
    min_p = None
    min_d = np.inf
    # Permutes over all predicted classes so to minimize the total distance
    for p in iter.permutations(range(U.shape[0])):
        d = np.sum([D[i, n] for n, i in enumerate(p)])
        if d < min_d:
            min_p, min_d = p, d
    return U[min_p, :]

# Plots a list on points on the given axis and colour them according to their
# selected class. If a desired classification Ud is provided then the legend
# will contain a sumary of the hits and misses for each class; misses are marked
# with an X while hits with Os. If centers C are provided, they are plotted
# with a black *
def plot(ax, X, U, Ud=None, C=None,
                 title="Clusters",
                 axes_names=["$X_0$", "$X_1$", "$X_2$"],
                 cluster_names=[]):
    C_N = U.shape[0]
    compare = Ud is not None
    error = 0

    # Creates the colormap to use
    cmap = plt.get_cmap("viridis")
    cnorm = mpl.colors.Normalize(vmin=0, vmax=C_N-1)
    ax.set_title(title)

    U = np.argmax(U, axis=0)
    Ud = np.argmax(Ud, axis=0) if compare else U

    for c in range(len(cluster_names), C_N):
        cluster_names.append(f"C{c}")

    for c in range(C_N):
        # Calculates hits and misses for the current class
        x = U == c
        xT = x & (Ud == c)
        xF = x & (Ud != c)
        t_c = np.sum(xT)
        c_name = cluster_names[c] + f": {t_c}"
        if compare:
            e_c = np.sum(xF)
            c_name += f"/{t_c+e_c} ({t_c/(t_c+e_c):.2%})"
            error += e_c
        # Plots hits and misses 
        ax.scatter(*X[:3, xT], color=cmap(cnorm(c)), label=c_name)
        ax.scatter(*X[:3, xF], color=cmap(cnorm(c)), marker="x")
        # If centers are provided, plot them with a black *
        if C is not None:
            ax.scatter(*C[:3, c], color=cmap(cnorm(c)),
                       marker="*", edgecolors="black", s=40)

    axis_labels(ax, *axes_names[:3])
    # Adds legend with total errors
    if compare:
        ax.scatter([], [], [], color='k', marker="x",
            label=f"Errors: {error}/{U.shape[0]} ({error/U.shape[0]:.2%})")
    ax.legend(loc="upper right")
