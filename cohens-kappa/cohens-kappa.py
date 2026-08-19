import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    
    # Write code here

    # convert input into numpy array
    rater1 = np.asarray(rater1, dtype=float)
    rater2 = np.asarray(rater2, dtype=float)

    # distict_labels1 = np.unique(rater1)
    # distinct_labels2 = np.unique(rater2)
    # distinct_label = distict_labels1 + distinct_labels2

    # we have better way to calculate unique values

    # # calculate po and pe
    # po = distinct_label / n
    # n = len(po)

    # pe = np.sum((distinct_labels1 * distinct_label) / (n * n))
    # numerator = po - pe
    # denominator = 1 - pe
    # k = numerator / denominator
    # return k

    # above is wrong way

    n = len(rater1)

    # find agreements, where both raters give the same label
    agreements = (rater1 == rater2)

    # po ==> actual agreement or observed agreement
    # pe ==> chance agreement

    # One sentence to remember:
    #
    # Agreement by chance is the agreement that could occur
    # simply because both raters have particular tendencies
    # to choose certain labels, even if their decisions are independent.

    number_of_agreements = np.sum(agreements)

    po = number_of_agreements / n

    # --------------------------------------------------
    # 2. Expected agreement by chance (Pe)
    # --------------------------------------------------

    # Find all possible labels
    labels = np.union1d(rater1, rater2)

    # union1d ==> Take the unique values from two arrays
    # and combine them into one array.

    """
    So NumPy gives us:

        labels = np.union1d(rater1, rater2)

        Mental model

        Rater 1 labels ──┐
                         ├──→ np.union1d() → ALL unique labels
        Rater 2 labels ──┘

        For Cohen's Kappa, this gives us the complete set
        of categories over which we calculate chance agreement.
    """

    # We initialize pe with 0 because we have to
    # accumulate chance agreement for every label.
    pe = 0

    for label in labels:

        # Probability that rater1 chooses this label
        p1 = np.sum(rater1 == label) / n

        # Probability that rater2 chooses this label
        p2 = np.sum(rater2 == label) / n

        # Probability that both independently choose this label
        pe += p1 * p2

    numerator = po - pe
    denominator = 1 - pe

    # If denominator is 0, then Pe = 1.
    # This happens when both raters use only one label.
    #
    # Example:
    # rater1 = [1, 1, 1, 1]
    # rater2 = [1, 1, 1, 1]
    #
    # In this case:
    # Po = 1
    # Pe = 1
    #
    # Normal formula becomes:
    # (1 - 1) / (1 - 1) = 0 / 0
    #
    # 0 / 0 is mathematically undefined,
    # so we explicitly return 1 for perfect agreement.
    if denominator == 0:
        return 1.0

    k = numerator / denominator

    return k

    """
    Now compare actual agreement with chance agreement

        We have:

        Po = 0.8

        and:

        Pe = 0.48

        Actual agreement:
        80%

        Expected agreement:
        48%

        So the agreement beyond chance is:

        0.8 - 0.48 = 0.32

        But why isn't Kappa simply 0.32?

        Because we need to normalize it.

        Kappa tells us how much of the possible
        beyond-chance agreement was actually achieved.

        κ = (Po - Pe) / (1 - Pe)

        κ = 1
        Perfect agreement.

        κ = 0
        No agreement beyond chance.
    """

    """

                Two raters
                  ↓
        Compare their labels
                  ↓
       ┌────────────────────┐
       │                    │
       ↓                    ↓
 Actual agreement      Label frequencies
       ↓                    ↓
      Po              Calculate probabilities
                            ↓
                           Pe
       └──────────┬─────────┘
                  ↓
       κ = (Po - Pe)/(1 - Pe)
                  ↓
              Kappa


    LABEL
      ↓
    Count how many times it occurs
      ↓
    FREQUENCY
      ↓
    Divide by total observations
      ↓
    PROBABILITY
      ↓
    Multiply Rater 1 probability × Rater 2 probability
      ↓
    Chance agreement for that label
      ↓
    Repeat for every label
      ↓
    Add them
      ↓
    Pe
    """

    """
        rater1, rater2
          ↓
          n
          ↓
    Compare positions
          ↓
    Observed agreement → Po
          ↓
    Find all labels
          ↓
    For each label:
        count label
          ↓
        convert count → probability
          ↓
        p1 × p2
          ↓
        add to Pe
          ↓
    Kappa = (Po - Pe)/(1 - Pe)
    """