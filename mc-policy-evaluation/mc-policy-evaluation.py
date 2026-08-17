import numpy as np


def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """

    """
    ================================================================
                        MONTE CARLO POLICY EVALUATION
    ================================================================

    Usually RL doesn't value a reward received 100 steps later
    exactly the same as a reward received immediately.

    So we introduce:

        γ = discount factor

    Where:

        0 ≤ γ ≤ 1


    The return becomes:

        Gt = Rt + γRt+1 + γ²Rt+2 + ...


    ================================================================
                    THE COMPLETE MONTE CARLO IDEA
    ================================================================

    We have a fixed policy:

        π


    We repeatedly generate complete episodes:

        Episode 1
        S → ... → Terminal
                  ↓
               calculate G


        Episode 2
        S → ... → Terminal
                  ↓
               calculate G


        Episode 3
        S → ... → Terminal
                  ↓
               calculate G


    Then:

        Value Estimate
             =
        Average of observed returns


    The complete flow is:

        Policy
           ↓
        Experience
           ↓
        Return
           ↓
        Value Estimate


    ================================================================
                    FIRST-VISIT VS EVERY-VISIT
    ================================================================

    First-visit Monte Carlo:

        For each episode, only use the first time
        a state appears.


    Example:

        A → B → A → C → Terminal

        For state A:

        Use first A
        Ignore second A


    Every-visit Monte Carlo:

        Use every occurrence of the state.

        A → B → A → C → Terminal
        ↑       ↑
        use     use
        both


    Both are ways of estimating:

        Vπ(s)


    ================================================================
                        THE KEY LIMITATION
    ================================================================

    Monte Carlo needs:

        complete episode


    It has to wait until the episode finishes before knowing
    the actual return.


    For example:

        S0 → S1 → S2 → S3 → Terminal
                             ↑
                        wait until here
                             ↓
                        calculate G


    It doesn't update V(S0) immediately.


    This leads naturally to Temporal-Difference (TD) Learning.


    TD says:

        "Why should I wait until the entire episode ends?
         I can learn from the next state immediately."


    That creates the famous distinction:


        Monte Carlo:

        State → ... → Terminal
                         ↓
                     actual return
                         ↓
                      update V



        TD:

        State → Next State
           ↓         ↓
        update using estimated future value


    This is one of the most important conceptual transitions
    in RL.


    ================================================================
                    THE LEARNING PROGRESSION
    ================================================================

        1. Agent / Environment
                ↓
        2. State
                ↓
        3. Action
                ↓
        4. Reward
                ↓
        5. Policy π
                ↓
        6. Return G
                ↓
        7. Value function Vπ(s)
                ↓
        8. Policy Evaluation
                ↓
        9. Monte Carlo Policy Evaluation
                ↓
        10. Temporal-Difference Learning
                ↓
        11. Q-function Qπ(s,a)
                ↓
        12. SARSA
                ↓
        13. Q-Learning
                ↓
        14. Policy Improvement
                ↓
        15. Policy Iteration
                ↓
        16. Actor-Critic


    The most important connection:

        Policy → Experience → Return → Value Estimate


    Then eventually:

        Value Estimate → Better Policy


    That second arrow is where policy improvement enters.


    ================================================================
                            THE KEY RULE
    ================================================================

    The problem represents an episode as:

        (state, reward)


    Example:

        [(0, 1), (1, 2), (2, 3)]

        state    reward
          ↓        ↓
        ( 0,       1 )
        ( 1,       2 )
        ( 2,       3 )


    The return recurrence is:

        Gt = rt + γGt+1


    If we process backward:

        G = reward + γG


    Example with γ = 1:

        (2, 3)

        G = 3


        (1, 2)

        G = 2 + 3
          = 5


        (0, 1)

        G = 1 + 5
          = 6


    Therefore:

        G0 = 6
        G1 = 5
        G2 = 3


    ================================================================
                    WHY PROCESS THE EPISODE BACKWARD?
    ================================================================

    Mathematical recurrence:

        Gt = rt + γGt+1


    Notice:

        To calculate Gt,
        we need the future return Gt+1.


    Therefore:

        If I process backward,
        I already know the future return.


    We can keep that future return inside one variable:

        G


    So:

        G = reward + gamma * G


    This lets us calculate every return in one backward pass.


    ================================================================
                    WHAT DO WE NEED TO STORE?
    ================================================================

    We need:

        returns_sum
        returns_count


    returns_sum:

        Stores the total first-visit returns collected
        for each state.


    returns_count:

        Stores how many first-visit returns we collected
        for each state.


    Therefore:

        V(state)
             =
        returns_sum[state] / returns_count[state]


    We also need to know the FIRST occurrence of each state.

    Why?

        Because we are processing backward.

    Example:

        Forward:

            A → B → A → C

        Backward:

            C → A → B → A
                ↑
                This is actually the SECOND A
                in forward time.


    Therefore, a simple:

        visited_states = set()

    during the backward traversal would select the wrong
    occurrence.

    Instead, we first record:

        first_indices[state]

    Example:

        A → B → A → C

        first_indices[A] = 0
        first_indices[B] = 1
        first_indices[C] = 3


    Then during the backward pass:

        if i == first_indices[state]

    means:

        "This is the first occurrence of this state
         in forward time."


    ================================================================
                        COMPLETE ALGORITHM
    ================================================================


                        EPISODE
                           │
                           ↓
                 Find first occurrence
                           │
                           ↓
                  Process backward
                           │
                           ↓
                    Read state,reward
                           │
                           ↓
                 G = reward + γ × G
                           │
                           ↓
                 Is this the first index?
                       ↙           ↘
                     YES           NO
                      ↓             ↓
                store return      ignore
                      │
                      ↓
             returns_sum[state]
                      │
                      ↓
            returns_count[state]
                      │
                      ↓
              Next episode
                      │
                      ↓
             sum[state] / count[state]
                      │
                      ↓
                       V


    ================================================================
                    WHY returns_sum AND returns_count?
    ================================================================

    Suppose state 0 appears in three different episodes:

        Episode 1 → return = 6
        Episode 2 → return = 10
        Episode 3 → return = 2


    Then:

        returns_sum[0]
            =
        6 + 10 + 2
            =
        18


        returns_count[0]
            =
        3


    Therefore:

        V[0]
            =
        18 / 3
            =
        6


    Important distinction:

        first_indices
            ↓
        Which occurrence should count
        inside ONE episode?


        returns_count
            ↓
        How many returns have been collected
        across ALL episodes?


    These solve different problems.


    ================================================================
    """

    """
    ================================================================
                    STEP 1: STORAGE
    ================================================================

    If:

        n_states = 3

    Then possible states are:

        0, 1, 2


    Therefore we need one storage position per state:

        State:          0      1      2
                         ↓      ↓      ↓
        returns_sum:   [0.0,   0.0,   0.0]


    returns_sum[state]
        =
    total returns collected for that state.
    """

    returns_sum = np.zeros(n_states)

    """
    returns_count stores how many first-visit returns
    were collected for each state.

        State:             0      1      2
                            ↓      ↓      ↓
        returns_count:    [0.0,   0.0,   0.0]


    Why?

        Average
            =
        Sum / Count


    Therefore:

        V[state]
            =
        returns_sum[state] /
        returns_count[state]
    """

    returns_count = np.zeros(n_states)


    """
    ================================================================
                    STEP 2: PROCESS EVERY EPISODE
    ================================================================
    """

    for episode in episodes:

        """
        ------------------------------------------------------------
        STEP 2A: FIND FIRST OCCURRENCE
        ------------------------------------------------------------

        We need the first occurrence in FORWARD order.

        Example:

            episode = [
                (0, 1),
                (1, 2),
                (0, -5),
                (2, 10)
            ]


        States:

            0 → 1 → 0 → 2


        First occurrence:

            state 0 → index 0
            state 1 → index 1
            state 2 → index 3
        """

        first_indices = {}

        for i, (state, reward) in enumerate(episode):

            if state not in first_indices:

                first_indices[state] = i


        """
        ------------------------------------------------------------
        STEP 2B: CALCULATE RETURNS BACKWARD
        ------------------------------------------------------------

        Start:

            G = 0


        Then process:

            last tuple
                 ↓
            previous tuple
                 ↓
            previous tuple
                 ↓
            ...
                 ↓
            first tuple


        Because:

            Gt = rt + γGt+1


        Code:

            G = reward + gamma * G
        """

        G = 0

        for i in range(len(episode) - 1, -1, -1):

            state, reward = episode[i]

            """
            Calculate current return.

            G already contains the future return.

                G = reward + gamma * G
            """

            G = reward + gamma * G


            """
            --------------------------------------------------------
            STEP 2C: FIRST-VISIT CHECK
            --------------------------------------------------------

            We are going backward.

            Therefore:

                i == first_indices[state]

            tells us whether this index is actually
            the FIRST occurrence in forward time.

            If YES:

                Store the return.

            If NO:

                Ignore this occurrence.
            """

            if i == first_indices[state]:

                returns_sum[state] += G

                returns_count[state] += 1


    """
    ================================================================
                    STEP 3: BUILD VALUE FUNCTION
    ================================================================

    Now we have:

        returns_sum[state]

    and:

        returns_count[state]


    Monte Carlo value:

        V(state)
            =
        average first-visit return


    Therefore:

        V(state)
            =
        returns_sum[state] /
        returns_count[state]


    States that were never visited should remain:

        0.0
    """

    V = np.zeros(n_states)

    for state in range(n_states):

        if returns_count[state] > 0:

            V[state] = returns_sum[state] / returns_count[state]


    """
    ================================================================
                            FINAL RESULT
    ================================================================

    V contains the estimated value of every state.

    Example:

        V = [6.0, 5.0, 3.0]


    Meaning:

        V[0] = estimated value of state 0
        V[1] = estimated value of state 1
        V[2] = estimated value of state 2
    """

    return V