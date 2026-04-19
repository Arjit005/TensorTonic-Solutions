import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x=np.array(x,dtype='float')
    mean=np.mean(x)
    median=np.median(x)
    freq=Counter(x)# gives output as dictionary 
    mode=max(freq,key=freq.get)
    return mean,median,mode
    