import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a=np.array(a)
    b=np.array(b)
    # requirement=1 => 1D Numpy array of equal length
    if len(a)==0 or len(b)==0:
        return 0.0
    elif len(a)==len(b):
        numerator=np.dot(a,b)
        magnitude_a=np.linalg.norm(a)
        magnitude_b=np.linalg.norm(b)
        if magnitude_a==0 or magnitude_b==0:
            return 0.0
        denominator=magnitude_a*magnitude_b
        ans=numerator/denominator
        return ans    
        
    