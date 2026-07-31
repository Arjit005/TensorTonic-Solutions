import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    # op is accuracy ,precision , recall , f1 score
    # Build confusion matrix then compute TP,TN,FN for each class

    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

    # concatinate to get unique classes
    classes=np.unique(np.concatenate((y_true,y_pred)))

    #np.concatenate() expects one argument: a sequence (tuple or list) of arrays.
    # because it is not only usng binary classification

    # calculate accuracy
    accuracy=np.mean(y_true==y_pred)

    # store metrics of every class
    precisions=[]
    recalls=[]
    f1_scores=[]
    supports=[]

    # needed for micro average
    total_TP=0
    total_FP=0
    total_FN=0

    # needed for binary average
    binary_precision=0
    binary_recall=0
    binary_f1=0

    for cls in classes:

        TP = TN = FP = FN = 0

        for a, p in zip(y_true, y_pred):

            if a == cls and p == cls:
                TP += 1

            elif a!=cls and p!=cls:
                TN=TN+1

            elif a!=cls and p==cls:
                FP=FP+1

            else:
                FN=FN+1

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP+FN ==0:
            recall=0
        else:
            recall=TP/(TP+FN)

        if precision+recall==0:
            f1=0
        else:
            f1=2*precision*recall/(precision+recall)

        # store every class metric
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # support = actual samples of this class
        supports.append(TP+FN)

        # for micro average
        total_TP+=TP
        total_FP+=FP
        total_FN+=FN

        # save binary metric
        if cls==pos_label:
            binary_precision=precision
            binary_recall=recall
            binary_f1=f1

    if average=="binary":

        precision=binary_precision
        recall=binary_recall
        f1=binary_f1

    elif average=="micro":

        if total_TP+total_FP==0:
            precision=0
        else:
            precision=total_TP/(total_TP+total_FP)

        if total_TP+total_FN==0:
            recall=0
        else:
            recall=total_TP/(total_TP+total_FN)

        if precision+recall==0:
            f1=0
        else:
            f1=2*precision*recall/(precision+recall)

    elif average=="macro":

        precision=np.mean(precisions)
        recall=np.mean(recalls)
        f1=np.mean(f1_scores)

    elif average=="weighted":

        supports=np.array(supports)
        weights=supports/np.sum(supports)

        precision=np.sum(weights*np.array(precisions))
        recall=np.sum(weights*np.array(recalls))
        f1=np.sum(weights*np.array(f1_scores))

    else:
        raise ValueError("Invalid average type")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }