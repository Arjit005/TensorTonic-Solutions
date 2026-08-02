def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0):

    if step > total_steps:
        return final_lr

    # Warmup
    if warmup_steps > 0 and step < warmup_steps:
        return (step * initial_lr) / warmup_steps

    # No warmup
    if warmup_steps == 0:
        if total_steps == 0:
            return final_lr
        return final_lr + (initial_lr - final_lr) * (
            (total_steps - step) / total_steps
        )

    # Linear decay
    return final_lr + (initial_lr - final_lr) * (
        (total_steps - step) / (total_steps - warmup_steps)
    )