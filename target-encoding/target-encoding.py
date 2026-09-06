def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    from collections import defaultdict
    
    # Compute sum and count for each category
    sums = defaultdict(float)
    counts = defaultdict(int)
    
    for cat, target in zip(categories, targets):
        sums[cat] += target
        counts[cat] += 1
    
    # Compute mean for each category
    means = {cat: sums[cat] / counts[cat] for cat in sums}
    
    # Replace each category with its mean
    return [means[cat] for cat in categories]