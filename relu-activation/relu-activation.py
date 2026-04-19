import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x=np.array(x)
    Relu=np.maximum(0,x)
    return Relu
    pass