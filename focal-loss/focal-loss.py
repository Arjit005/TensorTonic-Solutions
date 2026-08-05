import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    # convert input into numpy array
    p=np.asarray(p,dtype=float).ravel()
    y=np.asarray(y).ravel()
    # handling shape
    if p.shape!=y.shape:
        raise Value_Error("p and y must have the same shape")
    # prevent log zero
    p = np.clip(p, 1e-15, 1 - 1e-15)
        
    term_1=(1-p)**gamma * y * np.log(p)
    term_2=p**gamma * (1-y) * np.log(1-p)
     # calculating final loss
    focal_loss=-(term_1+term_2)
    return np.mean(focal_loss)