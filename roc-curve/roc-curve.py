import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.

    Returns:
        fpr, tpr, thresholds
    """

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by descending score
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    y_score = y_score[order]

    # Total positives and negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # Cumulative TP and FP
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    # Find last index of each unique score
    distinct = np.where(np.diff(y_score) != 0)[0]
    threshold_idx = np.r_[distinct, len(y_score) - 1]

    # Select only unique threshold points
    tp = tp[threshold_idx]
    fp = fp[threshold_idx]
    thresholds = y_score[threshold_idx]

    # Convert counts to rates
    tpr = tp / P if P > 0 else np.zeros_like(tp, dtype=float)
    fpr = fp / N if N > 0 else np.zeros_like(fp, dtype=float)

    # Add starting point
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds