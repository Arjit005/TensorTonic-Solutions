import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """

    """
    Binary Focal Loss
    -----------------

    We start from Binary Cross-Entropy (BCE):

        BCE = -log(pt)

    where pt means:

        "the probability that the model gave to the CORRECT class."

    For binary classification:

        target = 1  ->  pt = prediction
        target = 0  ->  pt = 1 - prediction

    Example:

        prediction = 0.9
        target = 1

        The model says:
            P(class 1) = 0.9

        Since the correct class is 1:

            pt = 0.9


        prediction = 0.1
        target = 0

        The model says:
            P(class 1) = 0.1

        Therefore:
            P(class 0) = 1 - 0.1 = 0.9

        Since the correct class is 0:

            pt = 0.9


    So the first thing we need to calculate is:

        pt = probability assigned to the correct class.
    """

    """
    Why do we need pt?
    ------------------

    Focal Loss is built on BCE.

    BCE is:

        BCE = -log(pt)

    If pt is HIGH:
        the model is confident about the correct answer
        -> easy example
        -> small BCE loss

    If pt is LOW:
        the model is not confident about the correct answer
        -> difficult example
        -> large BCE loss


    Example:

        pt = 0.9
        BCE = -log(0.9)
        BCE ≈ 0.105


        pt = 0.1
        BCE = -log(0.1)
        BCE ≈ 2.303


    Therefore:

        high pt -> easy example -> small loss
        low pt  -> hard example -> large loss


    But focal loss wants to go one step further.

    It asks:

        "Can we make easy examples contribute even LESS
         to the total loss?"

    This is especially useful when the dataset contains
    many easy examples.
    """

    """
    The focal factor
    -----------------

    Focal Loss adds:

        (1 - pt)^gamma

    to the BCE loss.

    Therefore:

        Focal Loss
        = - (1 - pt)^gamma * log(pt)


    Why does this work?

    Consider an EASY example:

        pt = 0.9
        gamma = 2

        (1 - pt)^gamma
        = (1 - 0.9)^2
        = 0.1^2
        = 0.01


    The BCE loss is therefore multiplied by 0.01.

    So the easy example is strongly downweighted.


    Now consider a HARD example:

        pt = 0.2
        gamma = 2

        (1 - pt)^gamma
        = (1 - 0.2)^2
        = 0.8^2
        = 0.64


    Now the loss is multiplied by 0.64.

    Therefore the hard example keeps much more of its loss.


    The pattern is:

        EASY example
             |
             | pt is high
             ↓
        1 - pt is small
             ↓
        focal factor is very small
             ↓
        loss is suppressed


        HARD example
             |
             | pt is low
             ↓
        1 - pt is large
             ↓
        focal factor is large
             ↓
        loss remains important
    """

    """
    What does gamma control?
    ------------------------

    gamma controls how strongly we focus on difficult examples.

    If:

        gamma = 0

    then:

        (1 - pt)^0 = 1

    Therefore:

        Focal Loss
        = -1 * log(pt)
        = BCE


    So:

        gamma = 0  -> ordinary BCE
        gamma > 0  -> easy examples are downweighted


    Example with pt = 0.9:

        gamma = 1:
            (1 - 0.9)^1 = 0.1

        gamma = 2:
            (1 - 0.9)^2 = 0.01

        gamma = 3:
            (1 - 0.9)^3 = 0.001


    As gamma increases, easy examples are suppressed more strongly.
    """

    """
    What does alpha do?
    -------------------

    Focal Loss can also use alpha:

        Loss
        = -alpha_t * (1 - pt)^gamma * log(pt)


    Alpha is used to give different importance to different classes.

    This is useful when the dataset is imbalanced.

    For example:

        95% -> class 0
         5% -> class 1

    Without class weighting, the majority class can dominate
    the training signal.

    Alpha allows us to assign different weights to the classes.


    Therefore remember:

        alpha  -> class balancing
        gamma  -> focus on difficult examples


    The complete equation is:

        L = -alpha_t * (1 - pt)^gamma * log(pt)
    """

    """
    Implementation plan
    -------------------

    We now have four things to calculate:

        1. Convert predictions and targets into arrays
        2. Calculate pt
        3. Calculate alpha_t
        4. Calculate focal loss
        5. Take the mean because the function asks for
           the MEAN binary focal loss


    The important part is step 2:

        pt depends on the target.

        target = 1 -> pt = prediction
        target = 0 -> pt = 1 - prediction


    We can express both cases together using:

        pt = targets * predictions
             + (1 - targets) * (1 - predictions)


    Why?

        If target = 1:

            pt = 1 * prediction + 0 * (1 - prediction)
               = prediction


        If target = 0:

            pt = 0 * prediction + 1 * (1 - prediction)
               = 1 - prediction


    Therefore this single expression gives us the
    probability of the correct class for every example.
    """

    # Write code here
    """
            predictions + targets
                ↓
              pt
                ↓
            alpha_t
                ↓
         focal factor
                ↓
             BCE part
                ↓
         multiply everything
                ↓
              mean
    """
    # convert array into numpy array
    predictions=np.asarray(predictions,dtype=float)
    targets=np.asarray(targets,dtype=float)
    # calculate probability 
        # if alpha>0 and gamma>=0:
        # if targets==1:
        #     p_t=predictions
        #     focal_loss=-alpha*(1-p_t)**gamma*np.log(p_t) 
        #     mean_FL= np.mean(focal_loss)
        #     return float(mean_FL)
        # elif targets==0:
        #     p_t=1-predictions
        #     focal_loss=-alpha*(1-p_t)**gamma*np.log(p_t)
        #     mean_FL= np.mean(focal_loss)

        # above is wrong way to handle bith cases we can do it in bettre way
    p_t=(
        targets*predictions+(1-targets)*(1-predictions)
        )

    log_pt=np.log(p_t) # LBCE
    #focal_loss for imbalance classes
    focal_loss=-((alpha*(1-p_t)**gamma)*log_pt)

    # calculate mean focal loss
    mean_FL=np.mean(focal_loss)
    return mean_FL
        