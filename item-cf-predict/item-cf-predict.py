def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    # Write code here
    # type hinting dekhkar ek chij pta chal rhi hai , output float chaiye
    # zero rating => item is unrated
    #Yaani items ke beech similarity.

    """
    Rating Matrix
      ↓
        Item-Item Similarity
              ↓
        User ne jo items rate kiye hain
              ↓
        Similar items identify karo
              ↓
        Weighted average
              ↓
        Predicted rating
    
    """
    #r_i is the user's rating for item i, and s_i 


    numerator = 0
    denominator = 0
    
    for i in range(len(user_ratings)):
        if i != target and user_ratings[i] != 0 and item_similarities[i] > 0:
            numerator += item_similarities[i] * user_ratings[i]
            denominator +=abs(item_similarities[i])
    if denominator==0:
        return 0.0
    res=numerator/denominator
    return float(res)