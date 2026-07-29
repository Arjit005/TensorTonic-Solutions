def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """

    new_values = []

    # Visit every state
    for s in range(len(values)):

        best_value = float("-inf")

        # Try every action
        for a in range(len(rewards[s])):

            future = 0

            # Compute Σ(T * V)
            for next_state in range(len(values)):
                future += transitions[s][a][next_state] * values[next_state]

            # Bellman update
            q = rewards[s][a] + gamma * future

            # Keep the best action
            best_value = max(best_value, q)

        new_values.append(float(best_value))

    return new_values