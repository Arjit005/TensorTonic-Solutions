import numpy as np
def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    # Write code here
    # first calculate weighted rating then apply it on every item and return list
    
    # first we need vote count, tab hm weighted rating formula lga payenge
    # convert items into np arrau 
    items=np.asarray(items)
    votes=items[:,-1]
    """
        matrix[:, -1]
    #      ↑   ↑
    #      │   └── last column
    #      └────── all rows
    """
    #calculate denominator
    shared_denominator=[]
    for i in range(len(votes)):
        
        total_votes=votes[i]+min_votes
        shared_denominator.append(total_votes)
    
    # compute average rting 
    average_rating=items[:,0]
    numerator1=votes*average_rating
    
    first_part=[]
    for i in range(len(numerator1)):
        first_one=numerator1[i]/shared_denominator[i]
        first_part.append(first_one)

    
    numerator2=global_mean*min_votes
    second_part=[]
    for i in range(len(numerator1)):
        sec_one=numerator2/shared_denominator[i]
        second_part.append(sec_one)

    weighted_rating=[]
    for i in range(len(first_part)):
        res=first_part[i]+second_part[i]
        weighted_rating.append(res)
    return weighted_rating