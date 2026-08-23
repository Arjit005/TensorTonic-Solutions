import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """

    #Expected Calibration Error (ECE) measures how far a model's confidence
    #is from its actual accuracy, averaged across probability bins.


    # converting input into np array
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)


    # create the bins dynamically:
    bins = [[] for _ in range(n_bins)]

    """
    then Python creates:

        [
            [],
            [],
            []
        ]

        Think:

        bins
         │
         ├── Bin 0 → []
         ├── Bin 1 → []
         └── Bin 2 → []

        Then we need to calculate which bin number a confidence belongs to.
    """

    # zip() pairs y_true and y_pred together
    #
    # y_true       y_pred
    #   ↓            ↓
    #   0          0.10
    #   0          0.20
    #   1          0.30
    #   1          0.40
    #
    # We need both values because later we compare
    # actual accuracy with predicted confidence.

    for actual, confidence in zip(y_true, y_pred):

        # Calculate which bin the confidence belongs to.
        #
        # Example:
        # confidence = 0.3
        # n_bins = 5
        #
        # 0.3 * 5 = 1.5
        # int(1.5) = 1
        #
        # Therefore confidence 0.3 goes into Bin 1.

        bin_index = min(
            int(confidence * n_bins),
            n_bins - 1
        )

        # Store both the actual value and confidence
        # inside the correct bin.
        bins[bin_index].append((actual, confidence))


    # Start ECE at zero
    ece = 0.0


    # Go through each bin one by one
    for b in bins:

        # If the bin is empty, skip it
        if not b:
            continue


        # Calculate actual accuracy inside this bin.
        #
        # Here we DON'T convert probability into 0 or 1.
        #
        # We simply calculate the average of y_true.
        #
        # Example:
        # y_true = [0, 1]
        #
        # accuracy = (0 + 1) / 2
        #          = 0.5

        acc = sum(
            y for y, p in b
        ) / len(b)


        # Calculate average confidence inside this bin.
        #
        # Example:
        # probabilities = [0.2, 0.3]
        #
        # confidence = (0.2 + 0.3) / 2
        #            = 0.25

        conf = sum(
            p for y, p in b
        ) / len(b)


        # Calculate how much of the total dataset
        # this bin represents.
        #
        # Example:
        # bin contains 2 predictions
        # total predictions = 10
        #
        # weight = 2 / 10
        #       = 0.2

        weight = len(b) / len(y_true)


        # Calculate this bin's calibration error:
        #
        # |actual accuracy - predicted confidence|
        #
        # Then multiply it by the bin's weight.

        ece += weight * abs(acc - conf)


    # Return the final Expected Calibration Error
    return ece