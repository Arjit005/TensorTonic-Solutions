import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    
    v=np.array(v)
    # case 1==> single 1 D vector
    if v.ndim==1:
        if len(v)!=3:
            return None
        magnitude=np.sqrt(np.sum(v**2))
        if magnitude==0:
            return None
        return v/magnitude 
    # handle multiple 3D vectors(NX3)
    elif v.ndim==2:
        if v.shape[1]!=3:
            return None
        magnitude=np.sqrt(np.sum(v**2,axis=1,keepdims=True))
         # avoid magnitude =0
        magnitude[magnitude==0]=1
        return v/magnitude
    else:
         return None