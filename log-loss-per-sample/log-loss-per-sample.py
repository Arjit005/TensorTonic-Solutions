import math
import numpy as np
def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    # binary  cross  entropy
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)

    """
    What is np.clip?
        np.clip is a NumPy function that limits values within a specified range.
        
        Any value lower than the minimum is set to the minimum.
        
        Any value higher than the maximum is set to the maximum.
        
        Values within the range remain unchanged.
        
        Syntax:
        
        python
        np.clip(array, min_value, max_value)
        Example:
        
        python
        import numpy as np
        
        arr = np.array([0.1, 0.5, 1.5, 2.0])
        clipped = np.clip(arr, 0.2, 1.0)
        print(clipped)  # [0.2 0.5 1.0 1.0]
        Here:
        
        0.1 was raised to 0.2 (the minimum).
        
        1.5 and 2.0 were lowered to 1.0 (the maximum).
    
    """
    p_hat = np.clip(y_pred, eps, 1 - eps)
    Log_loss=-(y_true*np.log(p_hat)+(1-y_true)*np.log(1-p_hat))

    return Log_loss.tolist()
    
    