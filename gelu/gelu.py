import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x=np.array(x,dtype='float')
    erf=np.vectorize(math.erf)
    cdf=(1+erf(x/(np.sqrt(2))))/2# erf is a function so it is taking arguments 
    Gelu=x*cdf
    return Gelu
