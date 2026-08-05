import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.

    Parameters:
        Z1 : np.ndarray of shape (N, D)
        Z2 : np.ndarray of shape (N, D)
        temperature : float

    Returns:
        float : InfoNCE loss
    """

    # Convert inputs to NumPy arrays
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    # Check shapes
    if Z1.shape != Z2.shape:
        raise ValueError("Z1 and Z2 must have the same shape")

    # Similarity matrix
    similarity = (Z1 @ Z2.T) / temperature

    # Numerical stability (row-wise)
    similarity = similarity - np.max(similarity, axis=1, keepdims=True)

    # Exponentiate
    exp_similarity = np.exp(similarity)

    # Numerator: positive pairs (diagonal)
    numerator = np.diag(exp_similarity)

    # Denominator: sum over each row
    denominator = np.sum(exp_similarity, axis=1)

    # Compute InfoNCE loss
    loss = -np.mean(np.log(numerator / denominator))

    return loss