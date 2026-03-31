import numpy as np

def softmax(x):
    x = np.array(x)

    # Case 1: 1D array
    if x.ndim == 1:
        diff = x - np.max(x)
        expo = np.exp(diff)
        return expo / np.sum(expo)

    # Case 2: 2D array (row-wise)
    elif x.ndim == 2:
        diff = x - np.max(x, axis=1, keepdims=True)
        expo = np.exp(diff)
        return expo / np.sum(expo, axis=1, keepdims=True)

    else:
        return None