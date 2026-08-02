import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code
    g = np.asarray(g, dtype=float)

    # handle invalid max_norm
    if max_norm <= 0:
        return g

    # compute global norm
    norm = np.linalg.norm(g)

    # handle zero norm
    if norm == 0:
        return g

    # No clipping needed
    if norm <= max_norm:
        return g

    # scale gradient
    # compute clipping or scale
    clip = max_norm / norm

    return g * clip