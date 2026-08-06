import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.asarray(X, dtype=float)

    # 1D array
    if X.ndim == 1:
        if strategy == "mean":
            value = np.nanmean(X)
        else:
            value = np.nanmedian(X)

        if np.isnan(value):
            value = 0.0

        X[np.isnan(X)] = value
        return X

    # 2D array
    if strategy == "mean":
        values = np.nanmean(X, axis=0) # column mean ,filling nan with column mean
    else:
        values = np.nanmedian(X, axis=0)

    # Replace NaN statistics (all-NaN columns) with 0
    values = np.nan_to_num(values, nan=0.0)

    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = values[cols]

    return X