import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    # it is the baseline in ML

    # converting input into np array
    y_train=np.asarray(y_train,dtype=int)
    X_test=np.array(X_test,dtype=int)

    # counting majority element 
    classes,counts=np.unique(y_train,return_counts=True)
    majority_classes=classes[np.argmax(counts)]
    # doing predictions
    predictions=np.full(len(X_test),majority_classes,dtype=int)

    return predictions
    

    
    
 