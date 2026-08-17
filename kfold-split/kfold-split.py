import numpy as np


def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns:
        list of length k with tuples (train_idx, val_idx)
    """

    """
    ============================================================
                    K-FOLD SPLIT — INDICES ONLY
    ============================================================

    K-Fold Split (Indices Only) means:

        Determine which row indices belong to training
        and which row indices belong to validation
        for each fold.

    We are NOT working with the actual X or y data here.

    We are only deciding:

        "Which indices should be TRAIN?"
        "Which indices should be VALIDATION?"

    That's why it is called:

                    K-Fold Split (Indices Only)
    """


    """
    ============================================================
                         STEP 1: CREATE INDICES
    ============================================================

    Suppose:

        N = 10 observations

    Then our dataset has these indices:

        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    np.arange(N) creates:

        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Visual:

        DATASET
        ─────────────────────────────────────────
        Index:    0   1   2   3   4   5   6   7   8   9
                  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
        Sample:   A   B   C   D   E   F   G   H   I   J
        ─────────────────────────────────────────
    """

    indices = np.arange(N)


    """
    ============================================================
                       STEP 2: OPTIONAL SHUFFLING
    ============================================================

    If shuffle=True, we shuffle the INDICES before
    creating the folds.

    Before:

        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    After shuffling, we might get:

        [4, 1, 8, 0, 6, 3, 9, 2, 7, 5]

    IMPORTANT:

        We are NOT changing the actual samples.

        We are only changing the order in which their
        indices will be divided into folds.

    Why?

        If the original data has some hidden ordering,
        shuffling prevents the folds from simply following
        that original order.

    If rng is provided:

        rng.permutation(indices)

    gives us a shuffled copy.

    If rng is not provided:

        np.random.shuffle(indices)

    shuffles the existing array.

    Visual:

        BEFORE SHUFFLING
        ─────────────────────────────────────────
        [0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
         │   │   │   │   │   │   │   │   │   │
        ─────────────────────────────────────────

                         ↓ SHUFFLE ↓

        AFTER SHUFFLING
        ─────────────────────────────────────────
        [4] [1] [8] [0] [6] [3] [9] [2] [7] [5]
        ─────────────────────────────────────────
    """

    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)


    """
    ============================================================
                    STEP 3: CREATE K FOLDS
    ============================================================

    Suppose:

        N = 10
        k = 5

    We want approximately equal-sized folds.

    Fold size:

        N / k

        10 / 5 = 2

    Therefore:

        Fold 0 → [0, 1]
        Fold 1 → [2, 3]
        Fold 2 → [4, 5]
        Fold 3 → [6, 7]
        Fold 4 → [8, 9]

    Visual:

        ORIGINAL INDICES
        ─────────────────────────────────────────
        [0  1  2  3  4  5  6  7  8  9]
        ─────────────────────────────────────────
          │     │     │     │     │
          ▼     ▼     ▼     ▼     ▼

        FOLD 0   FOLD 1   FOLD 2   FOLD 3   FOLD 4
        ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
        │ 0 1 │  │ 2 3 │  │ 4 5 │  │ 6 7 │  │ 8 9 │
        └─────┘  └─────┘  └─────┘  └─────┘  └─────┘

    np.array_split() does this division for us.

    ------------------------------------------------------------
    WHAT IF N IS NOT PERFECTLY DIVISIBLE BY K?
    ------------------------------------------------------------

    Example:

        N = 7
        k = 3

    7 / 3 = 2 remainder 1

    So we cannot make:

        2, 2, 2

    because that only gives 6 samples.

    One fold must receive the extra sample.

    Therefore:

        Fold 0 → [0, 1, 2]     ← 3 samples
        Fold 1 → [3, 4]        ← 2 samples
        Fold 2 → [5, 6]        ← 2 samples

    Visual:

        ┌─────────────┐
        │  0  1  2   │  ← 3
        └─────────────┘

        ┌─────────┐
        │  3  4   │  ← 2
        └─────────┘

        ┌─────────┐
        │  5  6   │  ← 2
        └─────────┘

    Fold sizes differ by at most 1.

    np.array_split() automatically handles this.
    """

    folds = np.array_split(indices, k)


    """
    ============================================================
              STEP 4: UNDERSTAND TRAINING / VALIDATION
    ============================================================

    Now the important idea:

        ONE fold → VALIDATION
        ALL OTHER FOLDS → TRAINING

    Suppose:

        Fold 0 → [0, 1]
        Fold 1 → [2, 3]
        Fold 2 → [4, 5]
        Fold 3 → [6, 7]
        Fold 4 → [8, 9]

    ------------------------------------------------------------
                         FOLD 0
    ------------------------------------------------------------

        Fold 0 → [0, 1]       ← VALIDATION

        Fold 1 → [2, 3]       ← TRAINING
        Fold 2 → [4, 5]       ← TRAINING
        Fold 3 → [6, 7]       ← TRAINING
        Fold 4 → [8, 9]       ← TRAINING

        Therefore:

            validation = [0, 1]

            training =
                [2, 3, 4, 5, 6, 7, 8, 9]


    ------------------------------------------------------------
                         FOLD 1
    ------------------------------------------------------------

        Fold 0 → [0, 1]       ← TRAINING
        Fold 1 → [2, 3]       ← VALIDATION
        Fold 2 → [4, 5]       ← TRAINING
        Fold 3 → [6, 7]       ← TRAINING
        Fold 4 → [8, 9]       ← TRAINING

        Therefore:

            validation = [2, 3]

            training =
                [0, 1, 4, 5, 6, 7, 8, 9]


    ------------------------------------------------------------
                         FOLD 2
    ------------------------------------------------------------

        Fold 0 → [0, 1]       ← TRAINING
        Fold 1 → [2, 3]       ← TRAINING
        Fold 2 → [4, 5]       ← VALIDATION
        Fold 3 → [6, 7]       ← TRAINING
        Fold 4 → [8, 9]       ← TRAINING


    ------------------------------------------------------------
                         FOLD 3
    ------------------------------------------------------------

        Fold 0 → [0, 1]       ← TRAINING
        Fold 1 → [2, 3]       ← TRAINING
        Fold 2 → [4, 5]       ← TRAINING
        Fold 3 → [6, 7]       ← VALIDATION
        Fold 4 → [8, 9]       ← TRAINING


    ------------------------------------------------------------
                         FOLD 4
    ------------------------------------------------------------

        Fold 0 → [0, 1]       ← TRAINING
        Fold 1 → [2, 3]       ← TRAINING
        Fold 2 → [4, 5]       ← TRAINING
        Fold 3 → [6, 7]       ← TRAINING
        Fold 4 → [8, 9]       ← VALIDATION


    ============================================================
                      THE CORE K-FOLD PATTERN
    ============================================================

                  F1       F2       F3       F4       F5
                ┌──────┬──────┬──────┬──────┬──────┐
    Iteration 1 │ VAL  │TRAIN │TRAIN │TRAIN │TRAIN │
                ├──────┼──────┼──────┼──────┼──────┤
    Iteration 2 │TRAIN │ VAL  │TRAIN │TRAIN │TRAIN │
                ├──────┼──────┼──────┼──────┼──────┤
    Iteration 3 │TRAIN │TRAIN │ VAL  │TRAIN │TRAIN │
                ├──────┼──────┼──────┼──────┼──────┤
    Iteration 4 │TRAIN │TRAIN │TRAIN │ VAL  │TRAIN │
                ├──────┼──────┼──────┼──────┼──────┤
    Iteration 5 │TRAIN │TRAIN │TRAIN │TRAIN │ VAL  │
                └──────┴──────┴──────┴──────┴──────┘

    The validation position keeps moving.

    This is the central idea of K-Fold.
    """


    result = []


    """
    ============================================================
                    STEP 5: LOOP THROUGH EACH FOLD
    ============================================================

    We now let every fold take one turn as validation.

    i = 0 → Fold 0 is validation
    i = 1 → Fold 1 is validation
    i = 2 → Fold 2 is validation
                     ...
    i = k-1 → Last fold is validation

    The model would normally be trained once for every
    iteration.

    Therefore:

        K folds → K training/evaluation experiments
    """

    for i in range(k):

        """
        --------------------------------------------------------
        CURRENT FOLD = VALIDATION
        --------------------------------------------------------

        If:

            i = 0

        then:

            val_idx = folds[0]

        If:

            i = 1

        then:

            val_idx = folds[1]

        And so on.
        """

        val_idx = folds[i]


        """
        --------------------------------------------------------
                     EVERYTHING ELSE = TRAINING
        --------------------------------------------------------

        We need every fold EXCEPT the current validation fold.

        Example:

            i = 1

            folds:

                Fold 0 → [0, 1]     ← TRAIN
                Fold 1 → [2, 3]     ← VAL
                Fold 2 → [4, 5]     ← TRAIN
                Fold 3 → [6, 7]     ← TRAIN
                Fold 4 → [8, 9]     ← TRAIN

        Therefore:

            train_idx =
                [0, 1, 4, 5, 6, 7, 8, 9]

        The list comprehension:

            [folds[j] for j in range(k) if j != i]

        means:

            "Give me every fold whose index is NOT i."

        np.concatenate() then joins those folds together.
        """

        train_idx = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )


        """
        --------------------------------------------------------
                         STORE THE RESULT
        --------------------------------------------------------

        We store:

            (train_idx, val_idx)

        So one element of result looks like:

            (
                [0, 1, 4, 5, 6, 7, 8, 9],
                [2, 3]
            )

        where:

            first  → training indices
            second → validation indices
        """

        result.append((train_idx, val_idx))


    """
    ============================================================
                         STEP 6: RETURN
    ============================================================

    After the loop finishes, we have K train/validation pairs.

    Visual:

        result
          │
          ├── Fold 0 → (TRAIN, VAL)
          │
          ├── Fold 1 → (TRAIN, VAL)
          │
          ├── Fold 2 → (TRAIN, VAL)
          │
          ├── Fold 3 → (TRAIN, VAL)
          │
          └── Fold 4 → (TRAIN, VAL)

    Every index appears exactly once in validation.

    Therefore:

        Each sample gets exactly one opportunity
        to be evaluated as unseen validation data.
    """

    return result

    """
                     K-FOLD CROSS VALIDATION

                         DATA
                          │
                          ▼
                  [0 1 2 3 4 5 ...]
                          │
                          ▼
                    SPLIT INTO K
                       FOLDS
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       Fold 1          Fold 2          Fold 3 ...
          │
          ▼
     One fold = VAL
     Others = TRAIN
          │
          ▼
     Train + Evaluate
          │
          ▼
     Move VAL → next fold
          │
          ▼
        Repeat K
        times
          │
          ▼
     Average scores
    """