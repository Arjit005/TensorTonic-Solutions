import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    sse=np.sum((y_true-y_pred)**2)
    y_mean=np.mean(y_true)
    sst=np.sum((y_true-y_mean)**2)
    if sst == 0:
        return 1.0 if sse == 0 else 0.0
    r2_score=1-(sse/sst)
    return r2_score
    pass