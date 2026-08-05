import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # convert input arrays into numpy array
    p=np.asarray(p,dtype=float)
    y=np.asarray(y)
    
    if p.shape != y.shape:
        raise ValueError("Shapes must match")
    numerator=2*np.sum(p*y)+eps
    denominator=np.sum(p)+np.sum(y)+eps
    dice=numerator/denominator

    dice_loss=1-dice
    return dice_loss