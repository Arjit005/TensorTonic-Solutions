import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct = np.sum(y_true == y_pred)# number of correct predictions
    # here FP==FN
    # and for multiclass==> TP+FP= total sample
    # so ,f1=correct prediction /total sample
    total = len(y_true)
    
    return correct / total