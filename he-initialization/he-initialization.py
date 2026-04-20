import math
def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    if fan_in<=0:
        raise ValueError("fan_in must be positive")
    limit=math.sqrt(6/fan_in)   
    factor = 2 * limit
    ans = [[(val * factor) - limit for val in row] for row in W]
    
    return ans