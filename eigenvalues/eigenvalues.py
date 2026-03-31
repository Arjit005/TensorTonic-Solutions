import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # requirement 1: fulfilled
    # matrix=np.asarray(matrix) #exception: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
    try:
        matrix = np.asarray(matrix, dtype=complex)
    except (ValueError, TypeError):
        return None

    # requirement 3: handle empty matrices
    if matrix.ndim != 2 or matrix.size == 0:
        return None

    # requirement 2 and 3: if matrix is not square return None
    if matrix.shape[0] != matrix.shape[1]:
        return None

    try:
        eigen_values = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # round to avoid floating point precision errors vs reference
    eigen_values = np.round(eigen_values, decimals=10)

    # skip sorting for single eigenvalue — lexsort raises tuple index out of range
    if eigen_values.size == 1:
        return eigen_values

    # np.lexsort() expects a single tuple/list of keys, not separate arguments.
    # last key = primary sort → sorts by real part first, imag as tiebreaker
    indices = np.lexsort((eigen_values.imag, eigen_values.real))
    return eigen_values[indices]