import numpy as np
def moving_median(values: list, window_size: int) -> list:
    """
    Returns the median of every complete sliding window.
    """
    # Write code here
    
    res=[]
    
    for i in range(len(values)-window_size+1):
        new_values=values[i : i + window_size]
        median_values=np.median(new_values)
        
        res.append(median_values)
    return res        
    