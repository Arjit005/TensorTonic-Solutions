import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    
    """
    # requirements:
    #  return a new numpy array
    # must not modify the original matrix
    # must work for non square matrices
    # don't use .T or np.transpose
    # use manual indexing with loops or array operations
    N=len(A)
    M=len(A[0])
    
    result_matrix=np.zeros((M,N))#This directly creates an M × N matrix filled with 0
    for i in range(N):
        for j in range(M):
            result_matrix[j][i]=A[i][j]
            
    return result_matrix
     
    
    """
    np.zeros(M,N)
    👉 This is wrong syntax

❌ Why?

np.zeros() expects one argument → a tuple (shape)

So Python thinks:

M = shape
N = dtype (wrong!)

👉 That’s why it will throw an error."""
