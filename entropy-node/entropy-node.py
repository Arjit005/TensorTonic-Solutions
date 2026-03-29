import numpy as np

def entropy_node(y):
    
    #Compute entropy for a single node using stable logarithms.
    
    """
    Entropy  fundamental concept which measures randomness in dataset.
    used as splitting critetion to build trees that maximize information gain

    Requirements
    Return single float value ≥ 0
    Handle empty nodes (return 0 entropy)
    Use stable logarithm computation (avoid log(0))
    Support multi-class problems
    Use base-2 logarithms for interpretability
    """
    if len(y)==0:
        return 0.0
    # problem => y is class label , we have to calculate probabilities
    # pura khel probability calulate krne ka hai 
    y=np.array(y)
    values,count=np.unique(y,return_counts=True)
    """It returns two arrays — that's why you unpack into two variables
    values — sorted array of unique elements
    counts — how many times each appears (same order as values)
    """
    probability=count/len(y)# count of class i /total samples
    Entropy=-np.sum(probability*(np.log2(probability)))
    return Entropy 