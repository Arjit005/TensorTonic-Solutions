import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here

    # convert arrays into numpy array
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    # validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")

    # Validate labels {-1,+1}
    if not np.all((y_true == -1) | (y_true == 1)):
        raise ValueError("y_true must contain only -1 and +1.")

    # compute y*s
    score = y_true * y_score

    # compute hinge loss
    loss = np.maximum(0, margin - score)

    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")