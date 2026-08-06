import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    # converting text into  numerical data
    #Correct way: One-Hot Encoding :Create one column for each category.
    # convert array into numpy array
    y=np.asarray(y,dtype=int)
    if num_classes is None:
        num_classes=np.max(y)+1
    if np.any(y>=num_classes) or np.any(y<0):
        raise ValueError("Labels are out of range")
    # create zero matrix
    Y=np.zeros((len(y),num_classes),dtype=float)
    # advance indexing
    Y[np.arange(len(y)),y]=1.0

    return Y
        