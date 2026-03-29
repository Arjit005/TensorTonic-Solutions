import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    
    x=np.array(x)
    p=np.array(p)
    shape_x=x.shape
    shape_p=p.shape
    prob_sum=np.sum(p)
    if shape_x==shape_p and np.isclose(prob_sum,1.0) :
        Expected_value=np.sum(x*p)
        return float(Expected_value)
     
    else:
        raise ValueError("Shapes must match and probabilities must sum to 1")
        
    
