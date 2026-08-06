import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    # convert input into np array
    X=np.asarray(X,dtype=int)
    y=np.asarray(y,dtype=int)

    # creating indecies 
    indices=np.arange(len(X))
    
    # shuffle them
    if rng is None:
        np.random.shuffle(indices)
    else:
        rng.shuffle(indices)
    #Use the shuffled indices to create shuffled copies.
    X = X[indices]
    y = y[indices]

    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        if drop_last and len(X_batch) < batch_size:
            continue    
    # instead of return use yield
        yield X_batch, y_batch#Give back one value, pause, then continue on the next request.
    