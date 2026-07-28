import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    # Step 1: Create all positions (column vector)
    pos = np.arange(seq_len)[:, None]

    # Step 2: Frequency indices
    freq_indices = np.arange((d_model + 1) // 2)

    # Step 3: Compute angles
    angles = pos / (base ** (2 * freq_indices / d_model))

    # Step 4: Compute sin and cos
    sin_val = np.sin(angles)
    cos_val = np.cos(angles)

    # Step 5: Create output matrix
    pe = np.zeros((seq_len, d_model), dtype=float)

    # Step 6: Fill alternating columns
    pe[:, 0::2] = sin_val
    pe[:, 1::2] = cos_val[:, :d_model // 2]

    return pe