import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    M=len(A)
    N=len(A[0])
    trace=0
    for i in range(M):
        for j in range(N):
            if i==j:
                trace=trace+A[i][j]
    return trace            
         
    pass
