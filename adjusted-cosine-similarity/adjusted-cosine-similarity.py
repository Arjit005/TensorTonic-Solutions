import numpy as np
def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    """
    Returns the adjusted cosine similarity between the requested items.
    """
    # Write code here
    # The basic idea:
    # Treat the ratings as vectors and measure the angle between them.
    #So adjusted cosine correctly recognizes that their preference patterns are the same.
    """
                 RAW RATINGS
                      ↓
               Calculate user average
                      ↓
           rating - user average
                      ↓
              Adjusted ratings
                      ↓
              Treat as vectors
                      ↓
             Cosine similarity
                      ↓
            Similarity between items
    
    """
    """
    [5, 3, 0]
     ↑  ↑  ↑
     │  │  └── not rated → ignore
     │  └───── rated → use
     └──────── rated → use 
    """
    #item_i and item_j tell us which two columns to compare
    numerator=0
    i_item_squared=0
    j_item_squared=0
    for i in range(len(ratings_matrix)):
        if ratings_matrix[i][item_i]==0 or  ratings_matrix[i][item_j]==0:
            continue
        row = np.array(ratings_matrix[i])
        user_mean=np.mean(row[row!=0])
        
        adjusted_i=ratings_matrix[i][item_i]-user_mean
        adjusted_j=ratings_matrix[i][item_j]-user_mean
        numerator+=(adjusted_i*adjusted_j)
        i_item_squared+=adjusted_i**2
        j_item_squared+=adjusted_j**2
    denominator=np.sqrt(i_item_squared)*np.sqrt(j_item_squared)
        
    if denominator==0:
        return 0.0
    sim=numerator/denominator
    return sim
        