import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """

    """
    ============================================================
    1. WHY DOES TRAIN-SERVING SKEW EXIST?
    ============================================================

    Train-serving skew exists because the model is trained in one
    environment and used in another, and the data-processing logic
    can accidentally become different between those two environments.
    """

    """
    But historically, ML systems often looked like this:

                         TRAINING
                            │
    Raw data ──→ Python preprocessing ──→ Model


                         SERVING
                            │
    User request ──→ Java preprocessing ──→ Model


    Different systems.

    Different languages.

    Different libraries.

    Different versions.

    Different developers.

    Different assumptions.


    Therefore:

        Training preprocessing ≠ Serving preprocessing

    can happen.
    """

    """
    ============================================================
    2. REALISTIC EXAMPLE: DIFFERENT PREPROCESSING
    ============================================================

    Suppose our model uses:

        age
        income


    During training, missing income values are replaced
    with the training median:

        income = income.fillna(training_median)


    Suppose:

        training median = ₹50,000


    So:

        missing income
                ↓
            ₹50,000


    But production accidentally uses:

        production median = ₹70,000


    Now:

        Training:
        missing → ₹50,000


        Serving:
        missing → ₹70,000


    The model sees something different from what it learned.

    That's skew.
    """

    """
    ============================================================
    3. ANOTHER EXAMPLE: ENCODING MISMATCH
    ============================================================

    The model doesn't understand our original meaning.

    It only sees numbers.


    Suppose during training:

        Delhi   → 1
        Mumbai  → 2
        Chennai → 3


    The model learns using those encoded values.


    But production accidentally uses:

        Delhi   → 3
        Mumbai  → 1
        Chennai → 2


    The model receives:

        Delhi → 3


    It may interpret that as Chennai.


    Nothing is wrong with the neural network.

    Nothing is wrong with the prediction code.

    The contract between preprocessing and model has been broken.
    """

    """
    ============================================================
    4. THE CORE PROBLEM
    ============================================================

        Training:

        raw data
             ↓
        preprocessing A
             ↓
        model


        Production:

        raw data
             ↓
        preprocessing B
             ↓
        same model


    So train-serving skew is an MLOps problem,
    not primarily a model-architecture problem.
    """

    """
    ============================================================
    5. WHY DO WE NEED TO DETECT IT?
    ============================================================

    Engineers repeatedly encounter situations like:

        "Model works perfectly offline."

                    ↓ deployment

        "Why is production performance terrible?"


    One possible answer:

        Training data ≠ Serving data
    """

    """
                         MODEL
                           ↑
              ┌────────────┴────────────┐
              │                         │
           TRAINING                 SERVING
              │                         │
        historical data             live data
              │                         │
        preprocessing              preprocessing
              │                         │
              └────── MUST MATCH ──────┘
    """

    """
    ============================================================
    6. WHAT THIS CODING PROBLEM GIVES US
    ============================================================

    In the real world:

        Raw data
            ↓
        Training preprocessing
            ↓
        Training features
            ↓
        distribution


        Raw data
            ↓
        Serving preprocessing
            ↓
        Serving features
            ↓
        distribution


    This problem gives us those distributions directly:

        train_dist
        serving_dist


    So we don't need to build the preprocessing pipelines.

    We only need to measure how different the two distributions are.
    """

    """
    ============================================================
    7. WHAT ARE WE COMPARING?
    ============================================================

    TRAIN                 SERVING

    0.10                    0.05
    0.20                    0.10
    0.30                    0.15
    0.25                    0.35
    0.15                    0.35
      │                       │
      └──────── compare ──────┘


    These are proportions inside bins.

    We compare corresponding bins:

        Training bin 1 ↔ Serving bin 1
        Training bin 2 ↔ Serving bin 2
        Training bin 3 ↔ Serving bin 3
        ...
    """

    """
    ============================================================
    8. WHY DO WE CALCULATE BIN-BY-BIN?
    ============================================================

    Because we want to answer:

        "How much did each part of the distribution move?"


    Example:

        Training       Serving
           0.10   →      0.05

        shifted.


    Another example:

        Training       Serving
           0.15   →      0.35

        shifted even more.


    PSI captures these individual shifts
    and combines them into one number.
    """

    """
    ============================================================
    9. OUR IMPLEMENTATION FLOW
    ============================================================

    train_dist
         │
         ↓
    for every feature
         │
         ├── training bins
         │
         ├── serving bins
         │
         ├── convert to NumPy
         │
         ├── add eps
         │
         ├── calculate PSI
         │
         ├── compare with threshold
         │
         └── save result
    """
    results = {} # because output is dict
    
    for feature in train_dist:
        train = np.array(train_dist[feature])
        serving = np.array(serving_dist[feature])
    
        train = train + eps
        serving = serving + eps
    
        difference = serving - train
        division = serving / train
    
        psi = np.sum(
            difference * np.log(division)
        )
    
        results[feature] = {
            'psi': float(psi),
            'skewed': bool(psi >= threshold)
        }
    
    return results
    