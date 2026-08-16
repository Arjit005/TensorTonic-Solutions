import numpy as np


def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """

    """
    ============================================================
    STEP 1 — WHAT IS A CONFUSION MATRIX?
    ============================================================

    A confusion matrix is a table counting how reality and
    prediction overlap.

                        Prediction
                      Pass        Fail

    Reality  Pass       80          20
             Fail       10          90


    Rows    → actual / true classes
    Columns → predicted classes
    """

    """
    ============================================================
    STEP 2 — CONVERT INPUTS TO NUMPY ARRAYS
    ============================================================

    We explicitly use int64.

    Why?

    np.asarray([]) normally creates:

        array([], dtype=float64)

    But np.bincount() needs integer indices.

    Therefore:

        dtype=np.int64

    makes even an empty input an integer array.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    """
    ============================================================
    STEP 3 — VALIDATE THE INPUT SHAPE
    ============================================================

    y_true and y_pred describe the same observations.

    Therefore they must have the same shape.

    Example:

        y_true = [0, 1, 1]
        y_pred = [0, 1, 0]

    Both contain 3 observations.
    """

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be 1-dimensional"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape"
        )

    """
    ============================================================
    STEP 4 — DETERMINE NUMBER OF CLASSES K
    ============================================================

    If num_classes is provided:

        K = num_classes

    Otherwise we infer K from the largest label.

    Example:

        labels = [0, 1, 2]

        largest label = 2

        K = 2 + 1 = 3


    Why +1?

    Because labels start from 0:

        0, 1, 2

    contains 3 classes.
    """

    if num_classes is None:

        """
        We cannot infer K from an empty array.

        Therefore the user must provide num_classes.
        """

        if len(y_true) == 0:
            raise ValueError(
                "num_classes must be provided when arrays are empty"
            )

        num_classes = int(
            max(y_true.max(), y_pred.max())
        ) + 1

    K = int(num_classes)

    if K <= 0:
        raise ValueError(
            "num_classes must be positive"
        )

    """
    ============================================================
    STEP 5 — VALIDATE LABELS
    ============================================================

    Valid labels must satisfy:

        0 <= label < K

    For K = 3:

        valid:
            0, 1, 2

        invalid:
            -1, 3, 4, ...
    """

    if len(y_true) > 0:

        if (
            np.any(y_true < 0)
            or np.any(y_true >= K)
            or np.any(y_pred < 0)
            or np.any(y_pred >= K)
        ):
            raise ValueError(
                "Labels must be in the range [0, num_classes - 1]"
            )

    """
    ============================================================
    STEP 6 — CONVERT (TRUE, PREDICTED) INTO ONE INTEGER
    ============================================================

    A confusion matrix cell is identified by:

        (true, predicted)

    But np.bincount() counts one-dimensional integers.

    Therefore we convert:

        (true, predicted)

    into:

        true × K + predicted


    Example:

        K = 2

        (0, 0) → 0 × 2 + 0 = 0
        (0, 1) → 0 × 2 + 1 = 1
        (1, 0) → 1 × 2 + 0 = 2
        (1, 1) → 1 × 2 + 1 = 3


    Every confusion-matrix cell gets a unique integer.
    """

    indices = y_true * K + y_pred

    """
    ============================================================
    STEP 7 — COUNT THE INDICES
    ============================================================

    Example:

        y_true = [0, 0, 1]
        y_pred = [1, 0, 1]

    Then:

        indices = [1, 0, 3]


    np.bincount() asks:

        "How many times does each number appear?"


        index:  0  1  2  3
        count:  1  1  0  1


    Therefore:

        [1, 1, 0, 1]
    """

    counts = np.bincount(
        indices,
        minlength=K * K
    )

    """
    ============================================================
    STEP 8 — RESHAPE INTO K × K
    ============================================================

    Suppose:

        K = 2

    We have:

        [1, 1, 0, 1]

    Reshape into:

        [[1, 1],
         [0, 1]]


    This is now our raw confusion matrix.
    """

    cm = counts.reshape(K, K)

    """
    ============================================================
    STEP 9 — NO NORMALIZATION
    ============================================================

    normalize='none'

    means:

        Return raw counts.

    Example:

        [[80, 20],
         [10, 90]]
    """

    if normalize == 'none':
        return cm.astype(int)

    """
    ============================================================
    STEP 10 — NORMALIZATION
    ============================================================

    The raw confusion matrix is already correct.

    Now we choose the denominator.

                        Denominator
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
            TRUE            PRED           ALL
              ↓              ↓              ↓
             ROWS          COLUMNS        TOTAL
    """

    if normalize == 'true':

        """
        ========================================================
        TRUE NORMALIZATION
        ========================================================

        Rows = actual classes.

        Therefore normalize each row.

        Example:

                    Prediction
                  Pass        Fail

        Actual Pass  80          20

        Row total:

            80 + 20 = 100


        Therefore:

            80 / 100 = 0.80
            20 / 100 = 0.20


        The row sums to:

            0.80 + 0.20 = 1.0
        """

        denominator = cm.sum(
            axis=1,
            keepdims=True
        )

    elif normalize == 'pred':

        """
        ========================================================
        PRED NORMALIZATION
        ========================================================

        Columns = predicted classes.

        Therefore normalize each column.

        Example:

                    Prediction
                  Pass        Fail

        Actual Pass  80          20
        Actual Fail  10          90


        Predicted Pass total:

            80 + 10 = 90


        So:

            80 / 90
            10 / 90
        """

        denominator = cm.sum(
            axis=0,
            keepdims=True
        )

    elif normalize == 'all':

        """
        ========================================================
        ALL NORMALIZATION
        ========================================================

        Divide every cell by the total number of observations.

        Example:

            [[80, 20],
             [10, 90]]

        Total:

            80 + 20 + 10 + 90 = 200


        Therefore:

            80 / 200
            20 / 200
            10 / 200
            90 / 200


        The entire matrix sums to 1.
        """

        denominator = cm.sum()

    else:

        raise ValueError(
            "normalize must be 'none', 'true', 'pred', or 'all'"
        )

    """
    ============================================================
    STEP 11 — HANDLE DIVISION BY ZERO
    ============================================================

    A class might have zero observations.

    Example:

        [[10, 5],
         [ 0, 0]]


    Second row:

        0 + 0 = 0


    We cannot calculate:

        0 / 0


    Therefore replace zero denominators with 1.

    This prevents division by zero.
    """

    denominator = np.where(
        denominator == 0,
        1,
        denominator
    )

    """
    ============================================================
    STEP 12 — FINAL NORMALIZATION
    ============================================================

    Divide every cell by the appropriate denominator.

        true → row total
        pred → column total
        all  → total matrix


    Convert to float because normalized values can be:

        0.0
        0.5
        0.75
        1.0
        ...
    """

    return cm.astype(float) / denominator