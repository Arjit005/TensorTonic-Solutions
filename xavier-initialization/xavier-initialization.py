import math
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    if fan_in <=0 or fan_out<=0:
        raise ValueError("Both values should be positive")
    limit=math.sqrt(6/(fan_in+fan_out))
    factor=2*limit
    ans=[[(val*factor)-limit for val in row ]for row in W]
    return ans