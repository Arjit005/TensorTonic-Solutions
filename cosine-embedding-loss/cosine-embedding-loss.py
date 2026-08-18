import math
import numpy as np


def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """

    # ---------------------------------------------------------
    # First of all, convert inputs into NumPy arrays.
    # ---------------------------------------------------------
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    numerator = np.dot(x1, x2)

    # denominator = math.sqrt(x1) * math.sqrt(x2)
    denominator = np.linalg.norm(x1) * np.linalg.norm(x2)

    cosine_similarity = numerator / denominator

    if label == 1:
        loss = 1 - cosine_similarity
        return float(loss)

    elif label == -1:
        loss = max(0, cosine_similarity - margin)
        return float(loss)

    """
    =============================================================
    YOUR CODE → PROBLEM → CORRECT IDEA
    =============================================================

    | Your code                 | Problem
    |---------------------------|--------------------------------
    | `x1 * x2`                 | Element-wise multiplication
    | `math.sqrt(x1)`           | `x1` is an array
    | `math.sqrt(x2)`           | `x2` is an array
    | `numerator / denominator` | Correct after fixing numerator
    | Positive-label loss       | Correct
    | Negative-label loss       | Correct

    Correct idea:

    | Your code                 | Correct idea
    |---------------------------|--------------------------------
    | `x1 * x2`                 | `np.dot(x1, x2)`
    | `math.sqrt(x1)`           | `np.linalg.norm(x1)`
    | `math.sqrt(x2)`           | `np.linalg.norm(x2)`
    | `numerator / denominator` | Keep it
    | Positive-label loss       | `1 - cosine_similarity`
    | Negative-label loss       | `max(0, cosine_similarity - margin)`


    =============================================================
    ARCHITECTURE
    =============================================================

                 Input vectors
                       ↓
              Convert to NumPy arrays
                       ↓
                  Dot product
                       ↓
                Vector magnitudes
                       ↓
              Cosine similarity
                       ↓
                  Check label
                 ↙           ↘
                1             -1
                ↓               ↓
             similar       dissimilar
                ↓               ↓
             1 - cos      max(0, cos-margin)
                ↓               ↓
                       Loss
    =============================================================
    """