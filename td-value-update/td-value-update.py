import numpy as np


def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns:
        Updated value function V_new
    """

    """
    ============================================================
    1. WHAT PROBLEM ARE WE SOLVING?
    ============================================================

    An agent moves from one state to another:

        State s  ── action ──>  State s_next
          |                         |
        V(s)                     V(s_next)

    V(s) means:

        "How valuable do I currently think state s is?"

    V(s_next) means:

        "How valuable do I currently think the next state is?"

    The goal of TD learning is to improve these value estimates
    as the agent gains experience.
    """

    """
    ============================================================
    2. THE KEY TD IDEA
    ============================================================

    We do NOT wait until the final outcome.

    Instead, after one transition:

        current state → next state

    we immediately use:

        reward + value of next state

    to improve the estimate of the current state.

    Example:

        A → B

        V(A) = 10
        V(B) = 20
        reward = 5

    We use:

        5 + γ(20)

    as information about how valuable A should be.
    """

    """
    ============================================================
    3. GAMMA (γ)
    ============================================================

    gamma is the discount factor.

        0 <= gamma <= 1

    It determines how much importance we give to future value.

    Example:

        gamma = 1

        Future value is fully considered.

        gamma = 0.5

        Only half of the next state's estimated value
        is considered.
    """

    """
    ============================================================
    4. TD TARGET
    ============================================================

    The TD target is:

        TD Target = r + γV(s_next)

    where:

        r          = reward received
        V(s_next)  = estimated value of the next state

    We are using the NEXT state's value to estimate
    how the CURRENT state's value should change.
    """

    """
    ============================================================
    5. TD ERROR
    ============================================================

    Suppose:

        V(A) = 10

    and:

        reward + γV(B) = 23

    Then:

        TD error = 23 - 10
                 = 13

    General formula:

        TD_error = r + γV(s_next) - V(s)

    TD error tells us how different our new target is
    from our current estimate.

    If:

        TD_error > 0

    the current value was too low.

    If:

        TD_error < 0

    the current value was too high.

    If:

        TD_error = 0

    no correction is needed.
    """

    """
    ============================================================
    6. ALPHA (α)
    ============================================================

    We do not usually replace V(s) completely with the TD target.

    Instead, we move only PART of the way toward the target.

    alpha is the learning rate.

        0 < alpha <= 1

    Larger alpha:
        Bigger update.

    Smaller alpha:
        Smaller update.
    """

    """
    ============================================================
    7. TD UPDATE
    ============================================================

    The update formula is:

        V(s) ← V(s) + α * TD_error

    Substituting TD_error:

        V(s) ← V(s) +
                α [r + γV(s_next) - V(s)]

    This is the one-step TD value update.
    """

    """
    ============================================================
    8. COPY THE INPUT
    ============================================================

    Convert V into a NumPy array and make a copy.

    Why copy?

    Suppose the caller gives us:

        V = [10, 20, 30]

    .copy() creates a separate array for this function.

    So:

        original V from caller
                 |
                copy
                 ↓
             local V

    We can now change local V without changing the original
    array outside the function.
    """

    V = np.asarray(V, dtype=float).copy()

    """
    ============================================================
    9. UNDERSTAND V[s] AND V[s_next]
    ============================================================

    Suppose:

        V = [10, 20, 30]

        s = 1
        s_next = 2

    Then:

        V[s]
        ↓
        V[1]
        ↓
        20

    So V[s] is the value of the CURRENT state.

    And:

        V[s_next]
        ↓
        V[2]
        ↓
        30

    So V[s_next] is the value of the NEXT state.

    IMPORTANT:

        V[s]       means "element at index s"

    It does NOT mean:

        V * s

    V * s would multiply the whole array by s.
    """

    """
    ============================================================
    10. CALCULATE TD ERROR
    ============================================================

    Formula:

        TD_error = r + γV(s_next) - V(s)

    Python:

        TD_error = r + gamma * V[s_next] - V[s]

    Example:

        r = 5
        gamma = 0.9
        V[s_next] = 30
        V[s] = 20

    Therefore:

        TD_error
        = 5 + 0.9 * 30 - 20
        = 5 + 27 - 20
        = 12
    """

    TD_error = r + gamma * V[s_next] - V[s]

    """
    ============================================================
    11. UPDATE THE CURRENT STATE
    ============================================================

    Formula:

        V(s) ← V(s) + α * TD_error

    Python:

        V[s] = V[s] + alpha * TD_error

    Example:

        V[s] = 20
        alpha = 0.1
        TD_error = 12

    Therefore:

        V[s]
        = 20 + 0.1 * 12
        = 20 + 1.2
        = 21.2

    Before:

        V = [10, 20, 30]

    After:

        V = [10, 21.2, 30]

    ONLY the current state's value changed.

    V[s_next] was used to calculate the error,
    but V[s_next] itself was NOT updated.
    """

    V[s] = V[s] + alpha * TD_error

    """
    ============================================================
    12. WHY RETURN V?
    ============================================================

    We did not create a separate array called V_new.

    Instead, we modified our local copy of V.

    Before update:

        V = [10, 20, 30]

    After update:

        V = [10, 21.2, 30]

    Therefore V now contains the UPDATED value function.

    So:

        return V

    is correct.

    We could also create:

        V_new = V.copy()

    and return V_new.

    But that is not necessary here because our local V
    already became the updated array.
    """

    return V