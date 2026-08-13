import numpy as np


def knn_distance(X_train, X_test, k):

    """
                     🎯 KNN GOAL

            "Find the k closest points"
                       │
                       ▼
              ┌────────────────┐
              │  X_train       │
              │  X_test        │
              │  k             │
              └────────────────┘
                       │
                       ▼
                Compare EVERY
                test ↔ train
                       │
                       ▼
                  Differences
                       │
                       ▼
              Square → Sum → √
                       │
                       ▼
                   DISTANCES
                   [1, 1, 3]
                       │
                       ▼
                    argsort
                       │
                       ▼
                SORTED INDICES
                   [0, 1, 2]
                       │
                       ▼
                     k = 2
                       │
                       ▼
                    [0, 1]
                       │
                       ▼
                  ✅ ANSWER


        🔄 REVERSE CHECK

             [0, 1]
                │
                ▼
          training indices
             │      │
             ▼      ▼
           X[0]    X[1]
             │      │
             ▼      ▼
             1       3
             │      │
             ▼      ▼
           |2-1|   |2-3|
             │      │
             ▼      ▼
             1       1
              \     /
               \   /
                \ /
                 ▼
          both are closest


        🧠 NUMPY FLOW

        X_test
           │
           ▼
      broadcasting
           │
           ▼
   pairwise differences
           │
           ▼
       distances
           │
           ▼
       argsort()
           │
           ▼
      first k indexes
           │
           ▼
        neighbors
    """

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    pairwise_distance = (
        X_test[:, np.newaxis, :]
        - X_train[np.newaxis, :, :]
    )

    distances = np.sqrt(
        np.sum(pairwise_distance ** 2, axis=2)
    )

    sorted_indices = np.argsort(distances, axis=1)

    n_train = X_train.shape[0]

    if k <= n_train:
        return sorted_indices[:, :k]

    ans = np.full(
        (X_test.shape[0], k),
        -1,
        dtype=int
    )

    ans[:, :n_train] = sorted_indices

    return ans