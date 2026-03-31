import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    # requirements: 1==> must work for numpy array or numpy lists
    x=np.array(x)
    y=np.array(y)
    distance=np.sqrt(np.sum((x-y)**2))
    return float(distance)
                     
    pass