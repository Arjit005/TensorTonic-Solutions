import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # convert input to numpy arrays
    real_scores=np.asarray(real_scores,dtype=float)
    fake_scores=np.asarray(fake_scores,dtype=float)
    output=np.mean(fake_scores)-np.mean(real_scores)
    return output