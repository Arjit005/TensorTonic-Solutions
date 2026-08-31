def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    # Write code here
    # measures fraction whose first k recommendations contain at least one relevant item 
    """
    
    Hit Rate@K basically poochta hai:

        Kitne users ko unki relevant item Top-K recommendations mein mil gayi?
        
        Formula:
        
        HitRate@K = {Number of users with at least one hit}}/ {Total number of users}} 


    K = 3 ka matlab

        Har user ki recommendation list mein se sirf pehli 3 items dekhenge.
        
        User 1 → [1, 5, 3]
        User 2 → [4, 8, 2]
        User 3 → [3, 6, 7]
        
        Baaki items:
        
        User 1 → 7, 2       ❌ ignore
        User 2 → 9, 1       ❌ ignore
        User 3 → 2, 5       ❌ ignore
    """
    """
    Requirements
        Inspect only the first k recommendations for each user.
        Count at most one hit per user.
        Divide the hit count by the number of users.
        Return 0.0 when there are no user.
        
    """
    # total _users=> rows in matrix
    
    total_user=len(recommendations)
    if total_user == 0:
        return 0.0

    hit=0
   
    for i in  range(len(recommendations)):
        for j in range(k):
            if recommendations[i][j]==ground_truth[i][0]: # because ground truth is 2D list
                hit+=1
                break
    hit_rate=hit/total_user

    return float(hit_rate)
    
