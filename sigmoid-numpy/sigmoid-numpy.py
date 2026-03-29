import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function. no python loops
    """
    
    # convert input array into numpy array
    # because numpy array are faster than simple array
    numpy_x=np.array(x)
    # activation function implementation
    activation_fun=1/(1+np.exp(-numpy_x))
    ans=np.asarray(activation_fun,dtype=float)
    return activation_fun

    
    
   