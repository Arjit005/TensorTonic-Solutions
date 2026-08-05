import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
   
    # convert inputs into np arrays
    p=np.asarray(p,dtype=float)
    q=np.asarray(q,dtype=float)
    # shape handling 
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    ## Prevent division by zero
    q_stable=q+eps

    # calculate diversion
    divergence=np.where(
        p>0,
        p*np.log(p/q_stable),
        0.0
        
    )
    # sum all of them 
    return np.sum(divergence)
    
    