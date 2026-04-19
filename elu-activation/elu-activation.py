
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    res=[]
    for x in x:
        if x>0:
            res.append(x)
        else:
            res.append(alpha*(math.exp(x)-1))
    return res         