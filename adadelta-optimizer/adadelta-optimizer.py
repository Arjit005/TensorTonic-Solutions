import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Convert inputs to NumPy arrays
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq, dtype=float)
    E_update_sq = np.asarray(E_update_sq, dtype=float)

    # Check shapes
    if w.shape != grad.shape:
        raise ValueError("w and grad must have the same shape")

    if w.shape != E_grad_sq.shape:
        raise ValueError("w and E_grad_sq must have the same shape")

    if w.shape != E_update_sq.shape:
        raise ValueError("w and E_update_sq must have the same shape")
        
    
    # rho is decay rate 
    new_E_grad_sq=(rho*E_grad_sq)+(1-rho)*(grad**2)
    #Compute parameter update
    delta_w=-(
    np.sqrt(E_update_sq+eps)/np.sqrt(new_E_grad_sq+eps)
    )*grad


    # Update Squared Update Average
    new_E_update_sq = rho * E_update_sq + (1 - rho) * (delta_w ** 2)
    # Update Parameters
    new_w = w + delta_w
    return (new_w, new_E_grad_sq, new_E_update_sq)