import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # coverting inputs into np array
    anchor=np.asarray(anchor,dtype=float)
    positive=np.asarray(positive,dtype=float)
    negative=np.asarray(negative,dtype=float)
    # handle reshaping 
    anchor = np.atleast_2d(np.asarray(anchor, dtype=float))
    positive = np.atleast_2d(np.asarray(positive, dtype=float))
    negative = np.atleast_2d(np.asarray(negative, dtype=float))

    # calculating distancew
    def d(x,y):
        return np.sum((x-y) ** 2,axis=1)
    loss=np.maximum(0,d(anchor,positive)-d(anchor,negative)+margin)

    return np.mean(loss)