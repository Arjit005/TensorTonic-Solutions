# import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    
    #Compute precision@k and recall@k for a recommendation list.
    
    # Write code here
    """ when we build recommendation system then we need 
    precision and    recall as a  evaluation metrics
    
precision==>measures what fraction of recommended items are  relevant .
    recall==> measures what fraction of relevant items we are recommnded.
    
    here recommendation list is top k list of recommended items
    
    precision=(top K intersection relevant)/k
    recall=(top k intersection relevant)/relevant
    """
    
    top_k=recommended[:k]
    count=0
    for i in range(len(top_k)):
        if top_k[i] in relevant:
            count=count+1
    precision=count/k     
    recall=count/len(relevant)
    res=[]
    res.append(precision)
    res.append(recall)
    return res