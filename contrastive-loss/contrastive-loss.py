import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    # convert arrays into np array
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y)

    # Handle (D,)
    if a.ndim == 1:
        a = a.reshape(1, -1)  # -1 select all columns automatically, 1 is for one row
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1.")

    # compute distance
    d = np.linalg.norm(a - b, axis=1)  # we are calculating across row

    # positive pair loss
    pos = y * d**2

    # negative pair loss
    neg = (1 - y) * (np.maximum(0, margin - d) ** 2)

    # total loss = pos + neg
    total_loss = pos + neg

    # Reduction
    if reduction == "mean":
        return np.mean(total_loss)
    elif reduction == "sum":
        return np.sum(total_loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")