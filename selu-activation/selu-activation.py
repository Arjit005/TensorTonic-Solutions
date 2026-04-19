import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    # Write code here
    x=np.array(x)
    res=[]
    for x in x:
        if x>0:
          res.append(lam*x)  
        else:
            res.append(lam*alpha*(np.exp(x)-1))
    return res
