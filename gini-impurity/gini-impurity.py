import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.

    The Decision Tree follows this process:

                        PARENT NODE
                             │
                             ▼
                     Try a possible split
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
             LEFT CHILD            RIGHT CHILD
                  │                     │
                  ▼                     ▼
            Count classes          Count classes
                  │                     │
                  ▼                     ▼
          Calculate class         Calculate class
           probabilities           probabilities
                  │                     │
                  ▼                     ▼
               Gini_L                Gini_R
                  │                     │
                  └──────────┬──────────┘
                             ▼
                     WEIGHT BY SIZE
                             │
                             ▼
                    WEIGHTED GINI
                             │
                             ▼
              Compare with other splits
                             │
                             ▼
                 Choose the lowest Gini
                             │
                             ▼
                        BEST SPLIT


    Gini impurity measures how mixed the classes are
    inside a single node.

                    Gini = 1 - Σ(pᵢ²)


    Example: A split that does not improve purity

    PARENT: [0, 0, 1, 1]
    Gini = 0.5

                PARENT
                   │
             ┌─────┴─────┐
             ▼           ▼
           LEFT        RIGHT
         [0, 1]        [0, 1]
          Gini=0.5      Gini=0.5

    Weighted Gini:

        (2/4) × 0.5 + (2/4) × 0.5
                     = 0.5

    Parent Gini  = 0.5
    Split Gini   = 0.5

    The split did not reduce impurity,
    so it provides no improvement.


    Example: A split that improves purity

    PARENT: [0, 0, 1, 1, 0, 1]
    Gini = 0.5

                     PARENT
                        │
                    split here
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
            LEFT                RIGHT
         [0, 0, 1]            [1, 0, 1]
         2 zeros, 1 one       1 zero, 2 ones

    Calculate Gini for each child,
    then weight each Gini by the
    proportion of samples in that child.

    Weighted Gini:

        (N_left / N_total) × Gini_left
        +
        (N_right / N_total) × Gini_right

                            │
                            ▼
                  Weighted Gini of split

    The Decision Tree compares the weighted
    Gini of different possible splits and
    chooses the split with the LOWEST value.

    Lower Gini
        ↓
    Less mixed
        ↓
    More pure
        ↓
    Better split
    """

    # Convert class labels into NumPy arrays
    y_left = np.asarray(y_left, dtype=float)
    y_right = np.asarray(y_right, dtype=float)

    # Number of samples in each child node
    N_L = len(y_left)
    N_R = len(y_right)

    # If both children are empty, there are no samples to calculate impurity
    if N_L == 0 and N_R == 0:
        return 0.0

    # If the left child is empty, calculate Gini using the right child
    if N_L == 0:
        right_classes, right_counts = np.unique(y_right, return_counts=True)
        right_probabilities = right_counts / N_R
        return 1 - np.sum(right_probabilities ** 2)

    # If the right child is empty, calculate Gini using the left child
    if N_R == 0:
        left_classes, left_counts = np.unique(y_left, return_counts=True)
        left_probabilities = left_counts / N_L
        return 1 - np.sum(left_probabilities ** 2)

    # Count how many samples belong to each class in each child node
    left_classes, left_counts = np.unique(y_left, return_counts=True)
    right_classes, right_counts = np.unique(y_right, return_counts=True)

    # Total number of samples after the split
    N = N_L + N_R

    # Convert class counts into class probabilities
    left_probabilities = left_counts / N_L
    right_probabilities = right_counts / N_R

    # Calculate Gini impurity of each child node
    # Gini = 1 - sum(probability^2)
    Gini_L = 1 - np.sum(left_probabilities ** 2)
    Gini_R = 1 - np.sum(right_probabilities ** 2)

    # Calculate weighted Gini impurity of the entire split
    # Each child's Gini is weighted by its proportion of the total samples
    Weighted_gini = (
        (N_L / N) * Gini_L
        + (N_R / N) * Gini_R
    )

    return Weighted_gini