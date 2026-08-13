import numpy as np


def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """

    """
                     🎯 GOAL
       preserve class proportions
                    │
                    ▼
              inspect y
                    │
                    ▼
             find classes
                    │
                    ▼
       ┌──────────────────────┐
       │ process each class   │
       └──────────────────────┘
                    │
                    ▼
            find class indices
                    │
                    ▼
              shuffle them
                    │
                    ▼
       calculate test_count
                    │
                    ▼
        ┌────────────────────┐
        │ keep ≥1 for train  │
        └────────────────────┘
                    │
                    ▼
             split indices
              /         \
             ▼           ▼
          TRAIN         TEST
             │           │
             └─────┬─────┘
                   ▼
          sort final indices
                   │
                   ▼
             X[index]
             y[index]
                   │
                   ▼
                OUTPUT
    """

    # Stratification means splitting the data while preserving
    # the original class distribution.

    # Tell me what classes I have and how many samples
    # each class contains, so I can stratify them.
    classes, count = np.unique(y, return_counts=True)

    X = np.asarray(X)
    y = np.asarray(y)

    train_indices = []
    test_indices = []

    for cls in classes:

        # Find the positions belonging to the current class
        indices = np.where(y == cls)[0]

        # Calculate how many samples from this class
        # should go into the test set
        n_test = round(len(indices) * test_size)

        # Keep at least one sample in training when possible
        if len(indices) > 1:
            n_test = min(n_test, len(indices) - 1)
        else:
            n_test = 0

        # Shuffle only the indices
        if rng is not None:
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        # Split the shuffled indices
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        # Save this class's indices
        test_indices.extend(test_idx)
        train_indices.extend(train_idx)

    # Reference expects final rows in original index order
    train_indices = np.sort(np.asarray(train_indices, dtype=int))
    test_indices = np.sort(np.asarray(test_indices, dtype=int))

    # Use the final indices to extract the actual data
    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test