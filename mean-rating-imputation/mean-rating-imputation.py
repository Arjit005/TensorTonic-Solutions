import numpy as np

def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    """
    Returns a copy with missing ratings replaced by user or item means.
    """

    # output list chaiye, with same shape 
    # zero ==> missing rating 
    # user mode ==> impute each zero with user mean, row wise mean 
    # item mode ==> impute each zero with item mean, col wise mean

    # Original matrix ko change nahi karna
    ratings_matrix = np.array(ratings_matrix, dtype=float).copy()

    # NaN ko 0 treat karenge
    ratings_matrix[np.isnan(ratings_matrix)] = 0

    if mode == "user":

        # calculate row wise mean, replace 0 with it
        for i in range(len(ratings_matrix)):

            row = ratings_matrix[i]

            # 0 ko mean mein include nahi karenge
            non_zero = row[row != 0]

            if len(non_zero) == 0:
                mean_user = 0.0
            else:
                mean_user = np.mean(non_zero)

            for j in range(len(ratings_matrix[0])):
                if ratings_matrix[i][j] == 0:
                    ratings_matrix[i][j] = mean_user

        return ratings_matrix.tolist()

    if mode == "item":

        # calculate col wise mean, replace 0 with it
        for i in range(len(ratings_matrix)):
            for j in range(len(ratings_matrix[0])):

                col = ratings_matrix[:, j]

                # 0 ko mean mein include nahi karenge
                non_zero = col[col != 0]

                if len(non_zero) == 0:
                    mean_items = 0.0
                else:
                    mean_items = np.mean(non_zero)

                if ratings_matrix[i][j] == 0:
                    ratings_matrix[i][j] = mean_items

        return ratings_matrix.tolist()