import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """

    # completeflow willl be ==>
    
    """
    C
    ↓
    Observed frequencies
    ↓
    Calculate row totals
    ↓
    Calculate column totals
    ↓
    Calculate grand total
    ↓
    Calculate Expected frequencies
    ↓
    Compare Observed vs Expected
    ↓
    (O - E)² / E
    ↓
    Add all cells
    ↓
    χ² statistic
    """
    # O=>observed 
    #E=> Expected

    # Convert input into NumPy array
    C = np.asarray(C, dtype=float)

    # Compute expected frequencies
    row_totals = C.sum(axis=1)
    col_totals = C.sum(axis=0)
    total = C.sum()

    expected = np.outer(row_totals, col_totals) / total

    # Compute chi-square
    chi2 = np.sum((C - expected) ** 2 / expected)

    return chi2, expected