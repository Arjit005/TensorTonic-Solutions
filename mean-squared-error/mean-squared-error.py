import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    predicted_val=np.array(y_pred)
    true_val=np.array(y_true)
    N=len(y_true)
    shape_predicted=predicted_val.shape
    shape_true=true_val.shape
    if shape_predicted== shape_true:
        MSE=(np.sum((predicted_val-true_val)**2))/N
        return MSE
    else: return None    
    
   
