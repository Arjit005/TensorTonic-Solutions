
import numpy as np

def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """

    # i have to calculate two things: mean and variance
    mean_values = np.mean(series)
    variance_value = np.var(series)

    # If the series is constant, variance = 0.
    # Normalized autocorrelation would become 0/0 → NaN.
    # By convention:
    # lag 0 → 1
    # all other lags → 0
    if variance_value == 0:
        return [1.0] + [0.0] * max_lag

    res = []

    for k in range(max_lag + 1):

        # Reset accumulator for every lag
        auto_variance = 0

        # Denominator is common for all i at this lag
        denominator = len(series) * variance_value

        for i in range(len(series) - k):

            part1 = series[i] - mean_values
            part2 = series[i + k] - mean_values

            numerator = part1 * part2

            # Accumulate all pair-products
            auto_variance += numerator

        # Normalize after accumulating all pairs
        auto_variance = auto_variance / denominator

        res.append(auto_variance)

    return res

