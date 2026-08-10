from math import inf
import numpy as np
"""
The Core Concept
Streaming Min-Max Normalization maintains running estimates of min and max (and optionally mean/variance) that update incrementally with each new data point—using O(1) memory and O(1) time per sample.



┌─────────────────────────────────────────────────────────────┐
│  RULE 1: UPDATE MEMORY FIRST                                │
│          state['min'] = min(old_min, batch_min)             │
│          state['max'] = max(old_max, batch_max)             │
│                                                             │
│  RULE 2: NORMALIZE USING THE NEW MEMORY                     │
│          output = (x - new_min) / (new_max - new_min + ε)   │
│                                                             │
│  NEVER normalize then update. ALWAYS update then normalize. │
└─────────────────────────────────────────────────────────────┘
"""



def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # initializing min max using dictionay
    return {
        'min':np.full(D,inf),
        'max':np.full(D,-inf)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Write code here
    X_batch=np.asarray(X_batch)
    state['min']=np.minimum(state['min'], X_batch.min(axis=0))
    state['max']=np.maximum(state['max'], X_batch.max(axis=0))
    #Normalize using UPDATED global statistics
    # np.maximum ensures range is at least eps, preventing div-by-zero
    range_val = np.maximum(state['max'] - state['min'], eps)
    return (X_batch- state['min']) / range_val