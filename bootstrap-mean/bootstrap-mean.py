import numpy as np


def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """

    """
    Bootstrap → creates a distribution of possible estimates
    → Confidence Interval → summarizes the uncertainty in the estimate.

    Bootstrap = repeatedly resample your data to understand
    how uncertain your estimate is.

    Estimate = your best guess of that value using available data.
    """

    """
        Original sample
        [40 50 60 70 80]
               │
               │ resample with replacement
               ↓
        ┌───────────────────────┐
        │                       │
        │ Bootstrap 1 → mean 58 │
        │ Bootstrap 2 → mean 66 │
        │ Bootstrap 3 → mean 58 │
        │ Bootstrap 4 → mean 62 │
        │ ...                   │
        │ Bootstrap 10000        │
        │                       │
        └───────────────────────┘
               ↓
        Distribution of means
    """

    """
    The confidence interval tells us about the uncertainty
    around that estimate.
    """

    """
                    ORIGINAL DATA
                         │
                         ▼
                 [40,50,60,70,80]
                         │
                         │
                    calculate
                         │
                         ▼
                   Estimate = 60
                         │
                         │
                "How uncertain is 60?"
                         │
                         ▼
              ┌─────────────────────┐
              │      BOOTSTRAP      │
              │                     │
              │ Resample repeatedly │
              │ with replacement    │
              └─────────────────────┘
                         │
                         ▼
              10,000 bootstrap means
                         │
                         ▼
             Bootstrap distribution
                         │
                         ▼
                 95% Confidence
                     Interval
    """

    """
    🧠 One sentence to remember

    Bootstrap is a technique where we repeatedly resample
    our existing sample with replacement to simulate how
    our estimate might vary if we collected new samples.

    And why we need it:

    To measure the uncertainty of our estimate when repeatedly
    collecting real samples isn't practical.

    The important chain is:

    Sample
        ↓
    Estimate
        ↓
    Bootstrap
        ↓
    Distribution of estimates
        ↓
    Confidence Interval
    """

    # ============================================================
    # STEP 1: Convert input into NumPy array
    # ============================================================

    # Convert x into a NumPy array.
    #
    # Example:
    #
    # x = [40, 50, 60, 70, 80]
    #
    # becomes:
    #
    # array([40., 50., 60., 70., 80.])

    x = np.asarray(x, dtype=float)


    # ============================================================
    # STEP 2: Create a random number generator
    # ============================================================

    # Bootstrap needs random sampling.
    #
    # If the user does not provide an RNG,
    # create one.

    if rng is None:
        rng = np.random.default_rng()


    # ============================================================
    # STEP 3: Create an empty list
    # ============================================================

    # This list will store the mean from
    # every bootstrap sample.
    #
    # Example:
    #
    # boot_means = []
    #
    # Later:
    #
    # boot_means = [58, 66, 58, 62, ...]

    boot_means = []


    """
            ┌──────────────────────┐
            │                      │
            ▼                      │
    Choose bootstrap sample        │
            ↓                      │
    Calculate mean                 │
            ↓                      │
    Store mean                     │
            │                      │
            └── repeat 1000 times ┘
    """


    # ============================================================
    # STEP 4: Repeat bootstrap process
    # ============================================================

    # n_bootstrap tells us how many times
    # we want to repeat the process.
    #
    # Default:
    #
    # n_bootstrap = 1000
    #
    # Therefore, this loop runs 1000 times.

    for _ in range(n_bootstrap):


        # --------------------------------------------------------
        # STEP 4A: Choose a bootstrap sample
        # --------------------------------------------------------

        # We randomly select values from x.
        #
        # size=len(x)
        # → choose the same number of values as x.
        #
        # replace=True
        # → a value can be selected more than once.
        #
        # Example:
        #
        # Original data:
        # [40, 50, 60, 70, 80]
        #
        # Bootstrap sample:
        # [40, 40, 60, 70, 80]
        #
        # 40 appears twice.
        # 50 was not selected.
        #
        # This is called:
        # "sampling with replacement"

        sample = rng.choice(
            x,
            size=len(x),
            replace=True
        )


        # --------------------------------------------------------
        # STEP 4B: Calculate the mean
        # --------------------------------------------------------

        # Example:
        #
        # sample = [40, 40, 60, 70, 80]
        #
        # mean = (40 + 40 + 60 + 70 + 80) / 5
        #      = 58

        mean = np.mean(sample)


        # --------------------------------------------------------
        # STEP 4C: Store the mean
        # --------------------------------------------------------

        # Store this bootstrap mean.
        #
        # After many iterations:
        #
        # boot_means = [
        #     58,
        #     66,
        #     58,
        #     62,
        #     ...
        # ]

        boot_means.append(mean)


    # ============================================================
    # STEP 5: Convert list into NumPy array
    # ============================================================

    # boot_means is currently a Python list.
    #
    # Convert it into a NumPy array so that
    # NumPy statistical functions can work with it easily.

    boot_means = np.asarray(boot_means)


    """
    Original data
    [40, 50, 60, 70, 80]
            │
            ▼
    Create bootstrap sample
    [40, 40, 60, 70, 80]
            │
            ▼
    Calculate mean
            │
            ▼
           58
            │
            ▼
    Store it in boot_means
            │
            ▼
    Repeat 1000 times
            │
            ▼
    1000 bootstrap means
            │
            ▼
    Bootstrap distribution
    """


    """
    One small correction to keep in mind 🧠

    The bootstrap distribution is not literally a collection
    of possible true population means.

    It is a distribution of estimates produced by resampling
    our observed data.

    We use its variability to quantify uncertainty about
    the population parameter.
    """


    # ============================================================
    # STEP 6: Calculate alpha
    # ============================================================

    # ci = confidence level.
    #
    # Example:
    #
    # ci = 0.95
    #
    # means we want a 95% confidence interval.
    #
    # alpha = 1 - ci
    #
    # alpha = 1 - 0.95
    #       = 0.05
    #
    # So 5% is outside our confidence interval.

    alpha = 1 - ci


    # ============================================================
    # STEP 7: Calculate lower boundary
    # ============================================================

    # For a 95% confidence interval:
    #
    # alpha = 0.05
    #
    # Divide the remaining 5% between
    # the two sides:
    #
    # 5% / 2 = 2.5%
    #
    # Therefore:
    #
    # Lower boundary = 2.5th percentile

    lower = np.percentile(
        boot_means,
        100 * alpha / 2
    )


    # ============================================================
    # STEP 8: Calculate upper boundary
    # ============================================================

    # Upper boundary is:
    #
    # 100% - 2.5%
    # = 97.5%
    #
    # Therefore:
    #
    # Upper boundary = 97.5th percentile

    upper = np.percentile(
        boot_means,
        100 * (1 - alpha / 2)
    )


    # ============================================================
    # STEP 9: Return the results
    # ============================================================

    # Return:
    #
    # boot_means → distribution of bootstrap means
    # lower      → lower confidence boundary
    # upper      → upper confidence boundary

    return boot_means, lower, upper