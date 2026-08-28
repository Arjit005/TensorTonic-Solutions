import numpy as np
def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    # Write code here
    
    #convert input into np array
    similarities=np.asarray(similarities,dtype=float)
    ratings=np.asarray(ratings)
    """
    
        Python mein iske liye ek natural tool hai:
        
        zip(similarities, ratings)
        
        🧠 Important
        
        Is problem ko solve karne ke liye NumPy ki zarurat nahi hai.
        
        Tum NumPy use kar sakte ho, but problem ka actual concept hai:
        
        parallel lists + filtering + weighted average
    """
    # similarity >0 and both lists should be same length 
    # we will use zip , it is a iterator
    # for similarity, rating in zip(similarities, ratings):
    numerator =0
    denominator=0
    for i in range(len(similarities)):
        if similarities[i]>0:
            numerator+=similarities[i]*ratings[i]
            denominator+=similarities[i]
    if denominator==0:
        return 0.0
    prediction=numerator/denominator  
    return round(prediction,6)
        