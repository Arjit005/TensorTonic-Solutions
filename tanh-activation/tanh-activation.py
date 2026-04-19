import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x=np.array(x)
    nemo=np.exp(x)-np.exp(-x)
    deno=np.exp(x)+np.exp(-x)
    tanh=nemo/deno
    return tanh