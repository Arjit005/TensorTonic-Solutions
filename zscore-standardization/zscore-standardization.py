import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    X=np.array(X,dtype=float)
    mean_u=np.mean(X,axis=axis,keepdims=True)
    std_u=np.std(X,axis=axis,keepdims=True)
    deno=std_u+eps
    
    return (X-mean_u)/deno
    