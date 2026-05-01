import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    # axis is user chosen , so keep axis=axis
    # keepdims for implicit broadcasting
    #        safe=np.where(Euclidean==0,1.0,Euclidean)  handle zero devision error

    if not isinstance(matrix, (list, np.ndarray)):
        return None
    try:
        matrix=np.asarray(matrix,dtype=float)
    except (ValueError, TypeError):
        return None
    if matrix.ndim != 2:
        return None 
    if axis is not None and axis not in (0, 1):
        return None    
    if norm_type=='l2':
        Euclidean=np.sqrt(np.sum(matrix**2,keepdims=True,axis=axis))
        safe=np.where(Euclidean==0,1.0,Euclidean)
        result=matrix/safe
        return result
        
    elif norm_type=='l1':
        Manhattan=np.sum(abs(matrix),axis=axis,keepdims=True)
        safe=np.where(Manhattan==0,1.0,Manhattan)
        result=matrix/safe
        return result
        
    elif norm_type=='max':
        max_Norm=np.max(np.abs(matrix),axis=axis,keepdims=True)
        safe=np.where(max_Norm==0,1.0,max_Norm)
        result=matrix/safe
        return result    
    
    else:
        return None
   