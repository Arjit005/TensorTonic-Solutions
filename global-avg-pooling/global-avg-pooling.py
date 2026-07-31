import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """

    # We kept ==> overall strength of the feature
    # GAP ==> we average entire feature map
    # x is feature map
    # we don't need exact locations
    # each channel becomes one number.
    # avg across Height and Width

    x = np.asarray(x, dtype=float)

    if x.ndim == 3:
        gap = np.mean(x, axis=(1, 2))
        return gap

    elif x.ndim == 4:
        gap = np.mean(x, axis=(2, 3))
        return gap

    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")