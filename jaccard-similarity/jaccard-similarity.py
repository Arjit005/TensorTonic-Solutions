import numpy as np
def jaccard_similarity(set_a: list, set_b: list) -> float:
    """
    Returns the Jaccard similarity of the two item collections.
    """
    # Write code here

    set_a=set(set_a)
    set_b=set(set_b)
    intersection_of_them=set_a.intersection(set_b)
    numerator=len(intersection_of_them)
    union_of_them=set_a.union(set_b)
    denominator=len(union_of_them)
    if numerator==0 or denominator==0:
        return 0.0
    
        
    jaccard_simil=numerator/denominator
    return float(jaccard_simil)