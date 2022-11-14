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

# Probability density functions with argument N. Generates x random realizations
pdf = {
    "Uniform": lambda x, N: sci.stats.uniform.pdf(x, 0, N),
    "Triangular": lambda x, N: sci.stats.triang.pdf(x, 0.5, -1, N+1),
    "Normal": lambda x, N: sci.stats.norm.pdf(x, N//2, N//6)
}

# Returns the CDF, self-information and entropy of a group of PDFs
def pdf_info(indices, pdf_names = None):
    if pdf_names is None:
        pdf_names = pdf.keys()
    
    pdf_set = {name: {} for name in pdf_names}
    for name in pdf_names:
        F = pdf[name](indices, len(indices))
        pdf_set[name]["F"] = F
        pdf_set[name]["I"] = np.log2(1/F)
        pdf_set[name]["H"] = np.sum(-F*np.log2(F))
    return pdf_set

# From an index column, samples 3 sets for training, testing, and validation
def sample(indices, sets=None, pdf_names=None):
    N_S = len(indices)
    if sets is None:
        sets = {"Train": N_S - int(0.2 * N_S) - int(0.2 * N_S),
                "Test": int(0.2 * N_S),
                "Validation": int(0.2 * N_S)}
    if pdf_names is None:
        pdf_names = pdf.keys()

    final_sample = {}
    for name in pdf_names:
        weights = pd.Series(pdf[name](indices, N_S), index=indices)
        sets_curr = {}
        
        for k, v in sets.items():
            sample = np.random.choice(
                weights.index, size=v, p=weights/weights.sum(), replace=False)
            sets_curr[k] = sample
            weights = weights.drop(sample)
        final_sample[name] = sets_curr
    return final_sample