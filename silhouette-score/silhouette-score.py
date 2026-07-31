import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # convert into numpy array
    X = np.asarray(X)
    labels = np.asarray(labels)

    # computing distance from every point to every point
    dist = np.linalg.norm(X[:, None] - X[None, :], axis=2)

    # We're going to calculate the silhouette score for all points.
    scores = []

    for i in range(len(X)):

        same_mask = labels == labels[i]
        same_mask[i] = False  # don't include itself

        a = np.mean(dist[i][same_mask])  # using boolean indexing and taking mean

        b = np.inf

        for cluster in np.unique(labels):

            # if it is in same cluster then skip it
            if cluster == labels[i]:
                continue

            mask = labels == cluster

            avg = np.mean(dist[i][mask])  # boolean indexing

            b = min(b, avg)

        # silhouette score for current point
        s = (b - a) / max(a, b)

        scores.append(s)

    return np.mean(scores)