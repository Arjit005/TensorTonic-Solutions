import math

def novelty_score(recommendations: list, item_counts: list, n_users: int) -> float:
    """
    Returns the average self-information of the recommended items.
    """
    # Write code here
    # output ==> hme float chaiye
    #c_i is item_counts[i]

    # Return the average novelty of the recommended items. Return 0.0 when recommendations is empty.
    if not recommendations:  # handle empty list
        return 0.0
    
    total_novelty = 0.0
    for item in recommendations:
        popularity = item_counts[item] / n_users
        novelty = -math.log2(popularity)
        total_novelty += novelty
    
    return total_novelty / len(recommendations)

    
    
    pass