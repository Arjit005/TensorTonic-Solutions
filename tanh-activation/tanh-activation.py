import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x=np.asarray(x,dtype=float)
    exp_positive=np.exp(x)
    exp_negative=np.exp(-x)
    
    result=(exp_positive-exp_negative)/(exp_positive+exp_negative)
    return result