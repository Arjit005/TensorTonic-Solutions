def top_k_recommendations(scores: list, rated_indices: list, k: int) -> list:
    """
    Returns the highest-scoring unrated item indices.
    """

    rated = set(rated_indices)

    pairs = [
        (score, index)
        for index, score in enumerate(scores)
        if index not in rated
    ]

    pairs.sort(key=lambda x: (-x[0], x[1]))

    return [index for score, index in pairs[:k]]