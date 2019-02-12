import numpy as np


def activation_dynamic(w, x, theta=0):
    value = 0
    for j in range(w.shape[0]):
        if ~np.isnan(w[j]):
            value += np.dot(w[j], x[j]) - theta

<<<<<<< HEAD
    value = np.sign(value)
    value[value == 0] = 1

    return value
=======
    if value >= 0:
        return 1
    else:
        return -1
>>>>>>> f08213942b74a5245ba514ebc452d2f7b31bd042
