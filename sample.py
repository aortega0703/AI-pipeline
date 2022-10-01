from ctypes import alignment
import scipy as sci
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def axis_labels(ax, x, y, z=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if z != None:
        ax.set_zlabel(z)

pdf = {
    "Uniform": lambda x, N: sci.stats.uniform.pdf(x, 0, N),
    "Triangular": lambda x, N: sci.stats.triang.pdf(x, 0.5, -1, N+1),
    "Normal": lambda x, N: sci.stats.norm.pdf(x, N//2, N//6)
}

def compare_PDF(indices, pdf_names = None):
    if pdf_names is None:
        pdf_names = pdf.keys()
    
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout(pad=2)
    fig.suptitle("Sample PDF analytical comparison", y=1.05)
    ax[0].set_title("Probability Density Function")
    ax[1].set_title("Information content")
    axis_labels(ax[0], "Sample Index", "$f(x)$")
    axis_labels(ax[1], "Sample Index", "$I(x)$")

    for name in pdf_names:
        F = pdf[name](indices, len(indices))
        I = np.log2(1/F)
        H = np.sum(-F*np.log2(F))
        ax[0].plot(indices, F)
        ax[1].plot(indices, I, label=f"{name}: {H:.2f}")
    fig.legend(title="Entropy $H(X)$", loc="lower center")


def compare_sample(indices, sets=None, pdf_names=None, name_S = "Uniform"):
    N_S = len(indices)
    if sets is None:
        sets = {"Train": N_S - int(0.2 * N_S) - int(0.2 * N_S),
                "Test": int(0.2 * N_S),
                "Validation": int(0.2 * N_S)}
    if pdf_names is None:
        pdf_names = pdf.keys()

    fig, ax = plt.subplots(1, 3)
    fig.tight_layout(pad=2)
    fig.suptitle("Sample PDF numerical comparison", y=1.05)
    for n, name in enumerate(pdf_names):
        weights = pd.Series(pdf[name](indices, N_S), index=indices)
        sets_curr = {}
        
        for k, v in sets.items():
            sample = np.random.choice(
                weights.index, size=v, p=weights/weights.sum(), replace=False)
            sets_curr[k] = sample
            weights = weights.drop(sample)
        ax[n].set_title(f"{name} Sampling")
        ax[n].hist(sets_curr.values(), stacked=True, bins=10)
        axis_labels(ax[n], 'Index', 'Sample Count')
        if name == name_S:
            final_sample = sets_curr
    legend = fig.legend([f"{k}: {len(v)}" for k, v in final_sample.items()]
                        + [f"Total: {N_S}"], loc="lower center")
    legend.legendHandles[-1].set_visible(False)
    return final_sample