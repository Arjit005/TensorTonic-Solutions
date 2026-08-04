import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    #Convert inputs to NumPy arrays first. Update velocity using momentum and learning rate
    w=np.asarray(w,dtype=float)
    v=np.asarray(v,dtype=float)
    grad=np.asarray(grad,dtype=float)

    # same shape as W
    v = v.reshape(w.shape)
    grad = grad.reshape(w.shape)

    # w_look=w-(momentum*(v))
    new_v=momentum*(v)+(lr*grad)
    new_w=w-new_v

    return (new_w,new_v)
