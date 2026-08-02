import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    #convert  x into numpy array
    X=np.asarray(X,dtype=float)
    N=X.shape[0]
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    X_centered=X-np.mean(X,axis=0)
    # compute covarience matrix
    cov = (X_centered.T @ X_centered) / (N - 1)

    # computing standard deviation
    std=np.std(X,axis=0,ddof=1)
    denominator_matrix=np.outer(std,std) #all pairwise products σi σj

    #pearson correlation
     # Pearson correlation
    with np.errstate(divide="ignore", invalid="ignore"): #simply says:"I know division by zero may happen. Don't print warnings. Just produce NaN."
        p = cov / denominator_matrix
    for i in range(len(std)):
        if std[i] > 0:
            p[i,i]=1.0
        else:
            p[i,i]=np.nan
    
    return p