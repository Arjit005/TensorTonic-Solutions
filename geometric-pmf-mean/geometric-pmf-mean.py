import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # concept is==>
    """
    GEOMETRIC DISTRIBUTION

Question:
"When will my first success happen?"

              ↓

       Choose probability p
              ↓
          p = 0.2
              ↓
    Failure probability = 0.8
              ↓
       ┌───────────────┐
       │ First success │
       └───────────────┘
              ↓
     Attempt 1 → 0.200
     Attempt 2 → 0.160
     Attempt 3 → 0.128
     Attempt 4 → 0.1024
     ...
              ↓
       Average attempt
              ↓
           1 / p = 5
    """
    # convert inouts into numpy arrays
    k=np.array(k)
    geometric_pmf=((1-p)**(k-1))*p
    

    mean=float(1/p)
    
    return geometric_pmf,mean