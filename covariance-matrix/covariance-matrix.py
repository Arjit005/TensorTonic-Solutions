import numpy as np

def covariance_matrix(X):

    # goal =>Compute covariance matrix from dataset X.
    # can't use np.cov
    X=np.array(X)
    if X.ndim!=2:
        return None

    N=X.shape[0]
    if N<2:
        return None
    
    u=np.mean(X,axis=0)
    X_centered=X-u
    X_transpose=X_centered.T
    #shape=X.shape
    
    mat_mul=np.dot(X_transpose,X_centered)     
    cov_matrix=mat_mul/(N-1)
    return cov_matrix
    
   
  