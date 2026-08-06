import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # coverting input to np array
    fpr=np.asarray(fpr,dtype=float)
    tpr=np.asarray(tpr,dtype=float)
    # validate length
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have the same length.")

    if len(fpr) < 2:
        raise ValueError("At least 2 points are required.")

    auc=np.trapezoid(tpr,fpr)
    return float(auc)