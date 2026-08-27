def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """

    """
    Har row = ek user.

    0 ka matlab:
    User ne item ko rate nahi kiya → ignore karo.
    """

    """
        for each row:

        1. non-zero ratings nikalo

        2. unka mean nikalo

        3. har element:
             agar 0 → 0
             warna → rating - mean
    """

    result = []

    for row in matrix:

        ratings = [x for x in row if x != 0]

        if ratings:
            mean = sum(ratings) / len(ratings)
        else:
            mean = 0

        # normalized = [
        #     0 if x == 0 else round(x - mean, 6)
        #     for x in row
        # ]
        normalized = []

        for x in row:
            if x == 0:
                normalized.append(0)
            else:
                normalized.append(round(x - mean, 6))

        result.append(normalized)

    return result