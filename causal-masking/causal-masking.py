import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    # score == attention matrix ,Rows = Query (the current word)
    #Columns = Key (the word it wants to look at)
    #triu = Triangular Upper
#    But what do we want to replace?
# We want to replace the future positions.
# Those are the upper triangular entries:

# So it's easier to create a mask of the positions to replace.
# np.triu(np.ones((3,3)), k=1)
# Output:

# [[0. 1. 1.]
#  [0. 0. 1.]
#  [0. 0. 0.]]
# The 1s are exactly the cells that should become -1e9.

#The original input has been changed, which violates the problem statement.that's why we create copy


#  convert scores into  numpy array
    scores=np.asarray(scores,dtype=float)
    # 
    T=scores.shape[-1]
    # create mask
    mask=np.triu(np.ones((T,T),dtype=bool),k=1)#Because we do not want to mask the diagonal. A token is allowed to attend to itself.
    masked_scores = scores.copy()
    masked_scores[..., mask] = mask_value
    return masked_scores