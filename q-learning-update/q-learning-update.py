import numpy as np


def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns:
        Updated Q-table Q_new
    """

    """
    ============================================================
    1. WHAT DOES Q STORE?
    ============================================================

    Q is a NumPy array (Q-table), not a Python dictionary.

    Think of Q like a table:

        Q = np.array([
            [2, 5, 1],
            [3, 8, 4],
            [6, 2, 7]
        ])

             action
             0      1      2

    state 0  2.0    5.0    1.0
    state 1  3.0    8.0    4.0
    state 2  6.0    2.0    7.0

    Q[s, a] means:

        "How valuable is it to take action a
         when the agent is in state s?"

    Unlike V[s], which stores one value per state,
    Q stores a value for EVERY state-action pair.
    """

    """
    ============================================================
    2. EXAMPLE Q-TABLE
    ============================================================

    If:

        s = 1
        a = 1

    then:

        Q[s, a]
        ↓
        Q[1, 1]
        ↓
        8.0

    This is the value of taking action 1 in state 1.
    """

    # Convert input into NumPy array.
    #
    # copy() creates a separate array so that the original
    # Q-table supplied by the caller is not modified.
    Q = np.asarray(Q, dtype=float).copy()

    """
    ============================================================
    3. WHAT HAPPENS AFTER TAKING ACTION a?
    ============================================================

    We are currently at:

        state s

    We take:

        action a

    We receive:

        reward r

    and arrive at:

        next state s_next

    Now Q-learning asks:

        "What is the BEST action I could take
         from this next state?"
    """

    """
    ============================================================
    4. WHAT IS s_next?
    ============================================================

    Suppose:

        current state = 1

    The agent takes some action and moves to:

        next state = 2

    Therefore:

        s_next = 2

    Now we want to know:

        "What are the Q-values of all possible actions
         from state 2?"
    """

    """
    ============================================================
    5. FIND THE BEST VALUE IN THE NEXT STATE
    ============================================================

    Q[s_next] gives us ALL action values for the next state.

    Example:

        Q[s_next] = [4, 9, 6]

    This means:

        action 0 → value 4
        action 1 → value 9
        action 2 → value 6

    The best possible next action has value:

        max(4, 9, 6) = 9

    In Python:

        np.max(Q[s_next])
    """

    best_next_value = np.max(Q[s_next])

    """
    ============================================================
    6. CALCULATE TD TARGET
    ============================================================

    Q-learning target:

        TD_target = r + gamma * max(Q[s_next])

    We take:

        immediate reward

    plus:

        discounted value of the BEST action
        available in the next state.


                         Q[s_next]
                             ↓

                    ┌───────┬───────┬───────┐
                    │   6   │   2   │   7   │
                    └───────┴───────┴───────┘
                        ↑       ↑       ↑
                     action0 action1 action2

                             ↓
                          np.max()

                             ↓

                              7
    """

    TD_target = r + gamma * best_next_value

    """
    ============================================================
    7. CALCULATE TD ERROR
    ============================================================

    TD_error tells us:

        "How different is the new target
         from our current estimate?"

    Formula:

        TD_error = TD_target - Q[s, a]
    """

    TD_error = TD_target - Q[s, a]

    """
    ============================================================
    8. UPDATE THE CURRENT STATE-ACTION PAIR
    ============================================================

    We update:

        Q[s, a]

    NOT the entire Q-table.

    Formula:

        Q(s,a) ← Q(s,a) + α * TD_error
    """

    Q[s, a] = Q[s, a] + alpha * TD_error

    """
    ============================================================
    9. RETURN THE UPDATED Q-TABLE
    ============================================================

    Q now contains the updated value.

    Therefore:

        return Q

    Remember:

        We changed Q[s, a]

        Q now IS the updated Q-table.
    """

    return Q