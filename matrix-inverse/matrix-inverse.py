import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A=np.array(A)
    # if matrix is non square then return None(requirement 2 : fullfill)
    if A.ndim!=2 or A.shape[0]!=A.shape[1]:
        return None
    
        
    #det=np.linalg.det(A)
    # singular matrix case
    # if abs(det)<1e-10:
    #     return None
    try:
       mat_inv=np.linalg.inv(A)
       return  mat_inv
    except np.linalg.LinAlgError: 
        return None
        
    
        

