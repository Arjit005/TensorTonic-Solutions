import numpy as np


def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns:
        Action index (int)
    """

    """
    ============================================================
    1. WHAT IS q_values?
    ============================================================

    q_values contains the Q-values of all possible actions
    for the CURRENT state.

    Example:

        q_values = [2, 5, 3]

    This means:

        action 0 → Q-value = 2
        action 1 → Q-value = 5
        action 2 → Q-value = 3

    The best action is action 1.
    """

    # Convert input into a NumPy array.
    q_values = np.asarray(q_values, dtype=float)

    """
    ============================================================
    2. WHAT IS EPSILON?
    ============================================================

    epsilon controls exploration.

        epsilon = 0

            Never explore.
            Always choose the best action.

        epsilon = 1

            Always explore.
            Always choose randomly.

        epsilon = 0.1

            Approximately 10% of the time:
                choose a random action.

            Approximately 90% of the time:
                choose the best action.
    """

    """
    ============================================================
    3. SET UP THE RANDOM NUMBER GENERATOR
    ============================================================

    If the caller did not provide rng:

        rng = None

    create our own random number generator.

    If the caller DID provide rng, we use that generator.
    """

    if rng is None:
        rng = np.random.default_rng()

    """
    ============================================================
    4. DECIDE: EXPLORE OR EXPLOIT?
    ============================================================

    Generate a random number between 0 and 1.

    Example:

        random_number = 0.15

    Suppose:

        epsilon = 0.20

    Since:

        0.15 < 0.20

    we explore.

    Otherwise, we exploit.
    """

    if rng.random() < epsilon:

        """
        ========================================================
        EXPLORE
        ========================================================

        Choose a random action.

        If q_values has length 3:

            actions = 0, 1, 2

        rng.integers(3) can return:

            0
            1
            2
        """

        action = rng.integers(len(q_values))

    else:

        """
        ========================================================
        EXPLOIT
        ========================================================

        Choose the action with the highest Q-value.

        Example:

            q_values = [2, 5, 3]

        np.argmax(q_values)

        returns:

            1

        because q_values[1] = 5 is the largest value.
        """

        action = np.argmax(q_values)

    """
    Return the selected action index.
    """

    return int(action)