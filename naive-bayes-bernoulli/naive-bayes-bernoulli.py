import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """

    """
    The entire Bernoulli Naive Bayes pipeline

    Think of it as:
        
        Training data
              ↓
        Estimate P(class)
              ↓
        Estimate P(feature = 1 | class)
              ↓
        For a new example:
              ↓
        Convert features to 0/1
              ↓
        Compute log prior
              ↓
        For every feature:
            x log θ
            + (1-x) log(1-θ)
              ↓
        Add all contributions
              ↓
        Get one score per class
              ↓
        Choose class with highest score
    """

    """
    Binary feature
      ↓
    P(x=1|class) = θ
    P(x=0|class) = 1-θ
          ↓
    one-feature formula
    θ^x(1-θ)^(1-x)
          ↓
    many features
    multiply them
          ↓
    Naive Bayes likelihood
    """

    # Convert inputs into NumPy arrays
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    # Find classes
    classes = np.unique(y_train)

    """
    We need the total number of training rows
    to calculate the denominator of P(y).

    total training rows
        ↓
    used for class prior
    """

    n_train = len(y_train)

    # ---------------------------------------------------------
    # STEP 1: Calculate class priors
    # ---------------------------------------------------------

    priors = []

    for c in classes:

        """
        For each class, we need:

        P(y=c) =
        number of times class c appears
        --------------------------------
        total number of training samples
        """

        class_count = np.sum(y_train == c)

        prior = class_count / n_train

        priors.append(prior)

    # Convert priors into log-space
    log_priors = np.log(priors)

    # ---------------------------------------------------------
    # STEP 2: Calculate theta
    # P(feature = 1 | class)
    # ---------------------------------------------------------

    thetas = []

    for c in classes:

        # Select only training rows belonging to class c
        mask = (y_train == c)

        class_X = X_train[mask]

        # Number of samples belonging to this class
        n_class = len(class_X)

        # Count how many times each feature is 1
        feature_ones = np.sum(class_X, axis=0)

        # Laplace smoothing
        theta = (feature_ones + 1) / (n_class + 2)

        thetas.append(theta)

    thetas = np.array(thetas)

    # ---------------------------------------------------------
    # STEP 3: Calculate log likelihood
    #
    # log P(x_i | y)
    #
    # = x_i log(theta)
    # + (1-x_i) log(1-theta)
    # ---------------------------------------------------------

    log_likelihood = (
        X_test[:, None, :] * np.log(thetas)[None, :, :]
        +
        (1 - X_test[:, None, :])
        * np.log(1 - thetas)[None, :, :]
    )

    # Add contributions from all features
    log_likelihood = np.sum(log_likelihood, axis=2)

    # ---------------------------------------------------------
    # STEP 4: Add log prior
    #
    # log P(y|x) ∝ log P(y) + log P(x|y)
    # ---------------------------------------------------------

    log_posterior = log_likelihood + log_priors

    return log_posterior