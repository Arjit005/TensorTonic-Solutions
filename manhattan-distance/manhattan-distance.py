import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    # mod of xi-yi
    x=np.array(x)
    y=np.array(y)
    distance=np.sum(np.abs(x-y))
    return float(distance)
    pass