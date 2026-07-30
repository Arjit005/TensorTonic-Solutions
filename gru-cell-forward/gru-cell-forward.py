import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """

    # Step 1: Convert everything to 2D
    x, was_1d = _as2d(x, params["Wz"].shape[0])

    # h_prev is previous history
    h_prev, _ = _as2d(h_prev, params["Uz"].shape[0])

    # Step 2: Update gate
    z = _sigmoid(
        x @ params["Wz"] +
        h_prev @ params["Uz"] +
        params["bz"]
    )

    # Step 3: Reset gate
    r = _sigmoid(
        x @ params["Wr"] +
        h_prev @ params["Ur"] +
        params["br"]
    )

    """
    Step 4:
    We don't want memory values to grow without bound.
    tanh compresses them into the range [-1, 1].

    "Based on the current input and the useful parts of the old memory,
    this is what the new memory could be."
    """
    h_candidate = np.tanh(
        x @ params["Wh"] +
        (r * h_prev) @ params["Uh"] +
        params["bh"]
    )

    # Step 5 new hidden state 
    h = (1 - z) * h_prev + z * h_candidate

    if was_1d:
        h = h[0]

    return h