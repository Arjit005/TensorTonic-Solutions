import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct = np.sum(y_true == y_pred)
    total = len(y_true)

    return correct / total