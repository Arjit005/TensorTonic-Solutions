import numpy as np


def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.

    Returns:
        (map_value, ap_per_query)
    """

    """
    ============================================================
    🧠 MAIN IDEA
    ============================================================

    For every query:

        y_score
           ↓
        Sort by score
           ↓
        Reorder y_true
           ↓
        Calculate precision at every rank
           ↓
        Keep precision where item is relevant
           ↓
        Calculate AP

    Then:

        AP for every query
              ↓
        Mean
              ↓
             mAP
    """

    # Store AP for every query
    ap_per_query = []


    """
    ============================================================
    🔁 PROCESS EACH QUERY
    ============================================================

    IMPORTANT:

    Different queries can contain different numbers of items.

    Example:

        Query 1 → 4 items
        Query 2 → 4 items
        Query 3 → 5 items

    Therefore, we convert EACH query into a NumPy array
    separately inside the loop.

    We do NOT convert the entire y_true_list into one
    NumPy array.
    """

    for y_true, y_score in zip(y_true_list, y_score_list):

        # Convert THIS query into NumPy arrays
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)


        """
        ========================================================
        1️⃣ SORT BY MODEL SCORE
        ========================================================

        Higher score → higher rank.

        np.argsort(-y_score)

        gives indices from highest score to lowest score.
        """

        order = np.argsort(-y_score)


        """
        ========================================================
        2️⃣ REORDER y_true
        ========================================================

        The ranking came from y_score.

        Now we must put y_true into the SAME ranking order.
        """

        y_true_sorted = y_true[order]


        """
        ========================================================
        3️⃣ COUNT TOTAL RELEVANT ITEMS
        ========================================================

        R = total relevant items in the ORIGINAL query.

        This is calculated BEFORE applying k.
        """

        total_relevant = np.sum(y_true)


        """
        ========================================================
        4️⃣ APPLY k CUTOFF
        ========================================================

        k = None
            → use all ranked items

        k = 2
            → use only top 2 ranked items
        """

        if k is not None:
            y_true_sorted = y_true_sorted[:k]


        """
        ========================================================
        5️⃣ COUNT RELEVANT ITEMS FOUND SO FAR
        ========================================================

        np.cumsum() gives the running number of relevant items.

        Example:

            y_true_sorted = [1,0,1,1]

            cumsum = [1,1,2,3]
        """

        cumulative_relevant = np.cumsum(y_true_sorted)


        """
        ========================================================
        6️⃣ CREATE RANKS
        ========================================================

        Example:

            4 items → [1,2,3,4]
            5 items → [1,2,3,4,5]
        """

        ranks = np.arange(1, len(y_true_sorted) + 1)


        """
        ========================================================
        7️⃣ CALCULATE PRECISION
        ========================================================

        Precision:

            relevant items found so far
            ───────────────────────────
            total items seen so far
        """

        precision = cumulative_relevant / ranks


        """
        ========================================================
        8️⃣ FIND RELEVANT POSITIONS
        ========================================================

        1 → relevant
        0 → not relevant

        Boolean mask:

            y_true_sorted == 1
        """

        relevant_mask = y_true_sorted == 1


        """
        ========================================================
        9️⃣ KEEP PRECISION ONLY AT RELEVANT POSITIONS
        ========================================================

        AP only uses precision where the current item
        is relevant.
        """

        relevant_precisions = precision[relevant_mask]


        """
        ========================================================
        🔟 CALCULATE AP
        ========================================================

        AP = sum of precision at relevant positions / R

        R is the total number of relevant items in the
        ORIGINAL query.
        """

        if total_relevant == 0:
            ap = 0.0
        else:
            ap = np.sum(relevant_precisions) / total_relevant


        """
        ========================================================
        1️⃣1️⃣ STORE AP
        ========================================================
        """

        ap_per_query.append(ap)


    """
    ============================================================
    1️⃣2️⃣ CALCULATE mAP
    ============================================================

    mAP = mean of AP values across all queries.
    """

    if len(ap_per_query) == 0:
        map_value = 0.0
    else:
        map_value = np.mean(ap_per_query)


    """
    ============================================================
    1️⃣3️⃣ RETURN
    ============================================================
    """

    return map_value, ap_per_query