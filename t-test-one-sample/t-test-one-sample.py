import numpy as np


def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """

    # Convert input into NumPy array
    x = np.asarray(x, dtype=float)

    # Number of observations
    n = len(x)

    # Calculate sample mean
    x_bar = np.mean(x)

    # Calculate differences from the sample mean
    diff = x - x_bar

    # Calculate sample standard deviation
    # ddof=1 because this is a SAMPLE standard deviation
    s = np.sqrt(np.sum(diff ** 2) / (n - 1))

    # Calculate t-statistic
    t = (x_bar - mu0) / (s / np.sqrt(n))

    return t