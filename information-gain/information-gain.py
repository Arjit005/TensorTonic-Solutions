import numpy as np

"""
                    PARENT DATASET
                 y = [0, 0, 1, 1]
                       │
                       │
                 Parent Entropy
                       │
                       ▼
              Choose a split rule
                 Age <= 30 ?
                       │
              ┌────────┴────────┐
              │                 │
           TRUE              FALSE
              │                 │
              ▼                 ▼
        LEFT CHILD         RIGHT CHILD
          [0, 0]              [1, 1]
              │                 │
              ▼                 ▼
       Entropy = 0        Entropy = 0
              │                 │
              └────────┬────────┘
                       │
                       ▼
             Weighted Child Entropy
                       │
                       ▼
              Information Gain
"""
def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]

    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """

    # Convert labels into NumPy array
    y = np.asarray(y, dtype=float)

    # Split the labels using the Boolean mask
    left_mask = y[split_mask]
    right_mask = y[~split_mask]

    # Number of samples in each child
    nL = len(left_mask)
    nR = len(right_mask)

    # Total number of samples
    N = nL + nR

    # Entropy of left and right child
    Entropy_left = _entropy(left_mask)
    Entropy_right = _entropy(right_mask)

    # Entropy before splitting
    Entropy_of_y = _entropy(y)

    # Computing Information Gain
    IG = Entropy_of_y - (
        (nL / N) * Entropy_left
        + (nR / N) * Entropy_right
    )

    return IG