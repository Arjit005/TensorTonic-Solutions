import numpy as np


def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """

    """
    ================================================================
                    ADVANTAGE COMPUTATION
                    REINFORCEMENT LEARNING
    ================================================================

    Since you just learned Monte Carlo Policy Evaluation,
    Advantage Computation builds directly on value functions.


    The key question changes from:

        "How good is this state?"


    to:

        "How much better or worse is this particular action
         than what I normally expect from this state?"


    That difference is called:

        ADVANTAGE


    Mathematical definition:

        A_t = G_t - V(s_t)


    where:

        G_t    = actual observed return
        V(s_t) = critic's expected value
    """


    """
    ================================================================
                    STEP 1: CONVERT INPUTS
    ================================================================

    Convert the inputs into NumPy arrays.

    states:

        tells us which state we were in at each timestep.


    rewards:

        tells us the reward received at each timestep.


    V:

        contains the value of every state.
    """

    states = np.asarray(states)
    rewards = np.asarray(rewards, dtype=float)
    V = np.asarray(V, dtype=float)


    """
    ================================================================
                    STEP 2: CREATE RETURN ARRAY
    ================================================================

    We need one return for every timestep.

    Example:

        states  = [0, 1, 2]

        returns = [0, 0, 0]


    Eventually:

        returns = [6, 5, 3]


    IMPORTANT:

        np.zeros() expects a SHAPE.

    Therefore:

        np.zeros(len(states))

    NOT:

        np.zeros(states)
    """

    returns = np.zeros(len(states))


    """
    ================================================================
                    STEP 3: CALCULATE RETURN G
    ================================================================

    Return:

        G_t = r_t + γG_(t+1)


    We process backward because the current return depends
    on the future return.


        timestep:

            0       1       2
            ↓       ↓       ↓
            r0      r1      r2


    Start from the end:

        G2 = r2

        G1 = r1 + γG2

        G0 = r0 + γG1


    We keep the future return inside:

        G
    """

    G = 0

    for i in range(len(states) - 1, -1, -1):

        G = rewards[i] + gamma * G

        returns[i] = G


    """
    ================================================================
                    STEP 4: COMPUTE ADVANTAGE
    ================================================================

    Mathematical definition:

        A_t = G_t - V(s_t)


    We already calculated:

        returns = [G0, G1, G2, ...]


    But there is an important detail.

    V contains the value of STATES:

        V[0] = value of state 0
        V[1] = value of state 1
        V[2] = value of state 2


    While states tells us which state occurred
    at each timestep.

    Example:

        states = [0, 1, 2]

        V[states]

             ↓

        [V[0], V[1], V[2]]

             ↓

        [0.5, 1.0, 1.5]


    Therefore:

        advantages = returns - V[states]
    """

    advantages = returns - V[states]


    """
    ================================================================
                            FINAL RESULT
    ================================================================

    Return one advantage value for every timestep.
    """

    return advantages


    """
    ================================================================
                    THE COMPLETE CONNECTION
    ================================================================


        Monte Carlo
             │
             ↓
        Observed return G_t
             │
             ↓
        How good was this action?
             │
             ↓
        Compare with V(s)
             │
             ↓
        Advantage
    """


    """
    ================================================================
                    V(s) VS Q(s,a)
    ================================================================


                         State S
                            │
                  ┌─────────┴─────────┐
                  ↓                   ↓
              Vπ(S)                  Qπ(S,A)
                  │                   │
                  │                   │
                  └─────────┬─────────┘
                            ↓
                     Qπ(S,A) - Vπ(S)
                            ↓
                     Aπ(S,A)
                            ↓
                  "Was action A good?"
    """


    """
    ================================================================
                    BUILDING THE FORMULA
    ================================================================


    How good is the state?

            ↓

          V(s)


    How good is this action?

            ↓

          Q(s,a)


    Compare them:

            ↓

        Q(s,a) - V(s)


    Therefore:

        A(s,a) = Q(s,a) - V(s)
    """


    """
    ================================================================
                        ONE STEP DEEPER
    ================================================================

    Now notice something interesting:

    We didn't actually calculate Q(s,a) directly.


    We used:

        G_t


    as an observed sample of:

        Q(s,a)


    Therefore:

        Q(s,a) ≈ G_t


    And:

        A(s,a) = Q(s,a) - V(s)


    becomes, from one Monte Carlo episode:

        A(s,a) ≈ G_t - V(s)


    This is the bridge:

        Monte Carlo returns
                ↓
        Advantage estimates
                ↓
        Policy Gradients
                ↓
        Actor-Critic


    This connection becomes very important later in RL.
    """