import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    # n=len of vector
    v=np.array(v)
    n=len(v)
    matrix=np.zeros((n,n))
    matrix=np.diag(v)
    return matrix
    pass
