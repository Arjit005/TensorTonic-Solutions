import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X=np.array(X)
    # x_min and x_max along axis
    x_min=np.min(X,axis=axis,keepdims=True)
    x_max=np.max(X,axis=axis,keepdims=True)
    
    denominator=np.maximum(x_max-x_min,eps)

    #scale and return 
    return (X-x_min)/denominator
    pass