import numpy as np

def matrix_factorization_sgd_step(U: list, V: list, r: float, lr: float, reg: float) -> list:
    """
    Returns the updated user and item vectors in a two-item list.
    """
    # Write code here
    # type hinting ko dekhkar ek chij samajh aa rhi hai ki yaha pr output list chaiye
    U=np.asarray(U,dtype=float)
    V=np.asarray(V,dtype=float)
    dot_prod_of_u_and_V=np.dot(U,V)
    prediction_error=r-dot_prod_of_u_and_V
    
    # updated U and V
    new_U=U+lr*(prediction_error*V-reg*U)
    new_V=V+lr*(prediction_error*U-reg*V)

    return [new_U,new_V]
    
    
    