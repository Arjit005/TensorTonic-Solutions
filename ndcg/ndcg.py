import math

def ndcg(relevance_scores: list, k: int) -> float:
    """Return normalized discounted cumulative gain at k."""

    # -------------------------
    # Calculate IDCG@k
    # -------------------------

    discount_comulated_gain = 0

    # Ideal ranking = relevance scores sorted from highest to lowest
    ideal_ranking = sorted(relevance_scores, reverse=True)

    # Calculate IDCG for top k
    for position, relevance in enumerate(ideal_ranking[:k], start=1):

        # Calculate gain
        gain = 2**relevance - 1

        # Calculate discount
        discount = math.log2(position + 1)

        # Accumulate gain
        discount_comulated_gain += gain / discount

    IDCG_k = discount_comulated_gain


    # -------------------------
    # Calculate DCG@k
    # -------------------------

    discount_comulated_gain = 0

    # Use the ORIGINAL ranking here
    for position, relevance in enumerate(relevance_scores[:k], start=1):

        # Calculate gain
        gain = 2**relevance - 1

        # Calculate discount
        discount = math.log2(position + 1)

        # Accumulate gain
        discount_comulated_gain += gain / discount

    DCG_k = discount_comulated_gain


    # -------------------------
    # Calculate NDCG@k
    # -------------------------

    if IDCG_k == 0:
        return 0.0

    return DCG_k / IDCG_k