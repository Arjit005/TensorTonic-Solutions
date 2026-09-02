def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """

    # type hinting suggests that output will be a list


    """
    🚨 Problem with SMA

    SMA treats every value inside the window equally.

    For:

    10, 12, 9

    The weights are:

    10 → 1/3
    12 → 1/3
     9 → 1/3

    But intuitively, shouldn't the most recent value matter more?

    That's where EMA comes in.
    """


    """
    🧠 Core idea of EMA

    New EMA = some influence from the current value
            + some influence from the previous EMA

    EMA gives more importance to recent values while still
    carrying information from the past through the previous EMA.
    """


    """
    SMA:

    current value
    previous values
    previous values
    previous values
           ↓
    calculate average


    EMA:

    previous EMA
         +
    current value
         ↓
    new EMA

    So EMA is recursive.
    """


    """
    📌 Important idea

    Recent observations should have more influence than
    older observations.

    Conceptually, the weights become:

    Current value  → highest weight
    Older value    → smaller weight
    Older value    → even smaller weight
    Older value    → even smaller weight

    This decreasing influence is why it is called
    "Exponential" Moving Average.
    """


    """
    🧮 Initial EMA

    We need a starting point.

    EMA_zero = values[0]

    Example:

    values = [10, 20, 30]

    EMA_0 = 10

    We store the initial EMA in our result list.
    """

    EMA_zero = values[0]

    res = []
    res.append(EMA_zero)


    """
    🔄 Calculate the remaining EMA values

    We start from index 1 because values[0] is already
    used as the initial EMA.

    For every new value:

    current value  → values[i]
    previous EMA   → res[i - 1]

    Formula:

    EMA_t = alpha * current value
          + (1 - alpha) * previous EMA

    Therefore:

    EMA_t = alpha * values[i]
          + (1 - alpha) * res[i - 1]

    After calculating the new EMA, we append it to res.

    This creates the chain:

    EMA_0
      ↓
    EMA_1
      ↓
    EMA_2
      ↓
    EMA_3
      ↓
      ...
    """

    for i in range(1, len(values)):
        EMA_t = alpha * values[i] + (1 - alpha) * res[i - 1]
        res.append(EMA_t)

    return res

