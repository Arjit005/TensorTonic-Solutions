import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """

    # X_hat is simply the normalized version of X.
    # Gamma is a scaling knob.
    # Beta (β) ==> After scaling, maybe we want to shift everything.

    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        # (N, D)

        # computing mean
        mean = np.mean(x, axis=0)

        # computing variance
        var = np.var(x, axis=0)

        # Normalize every value
        X_hat = (x - mean) / np.sqrt(var + eps)

        # result
        res = gamma * X_hat + beta

    elif x.ndim == 4:
        # (N, C, H, W)

        # computing mean for each channel
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)

        # computing variance for each channel
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        # Normalize every value
        X_hat = (x - mean) / np.sqrt(var + eps)

        # reshape gamma and beta for broadcasting
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        # result
        res = gamma * X_hat + beta

    else:
        raise ValueError("Input must be 2D or 4D")

    return res