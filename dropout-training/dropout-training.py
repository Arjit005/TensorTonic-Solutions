import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    # size of random number generator= same as x
    x=np.asarray(x,dtype=float)
    if rng is not None:
        rand=rng.random(x.shape)
    else:
        rand=np.random.random(x.shape)
    keeping_prob=1.0-p
    scale=1.0/keeping_prob
    # mask store which is dropped and which not , for back propogation
    Keep_mask=rand<keeping_prob# comparing value  output,true,and false
    output_pattern=Keep_mask.astype(float)*scale
    output=x*output_pattern
   
    return (output,output_pattern)
    