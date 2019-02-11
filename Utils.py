import numpy as np


def activation_dynamic(w, x, theta=0):
    value = 0
    for j in range(w.shape[0]):
        if ~np.isnan(w[j]):
            value += np.dot(w[j], x[j]) - theta

    return np.sign(value)
