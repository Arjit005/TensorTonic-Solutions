import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """

    # complete work-flow will be ==>
    
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



What code is doing==>

                 C
                 ↓
          OBSERVED DATA
                 ↓
      Calculate row/column totals
                 ↓
        Calculate EXPECTED
                 ↓
       Compare O with E
                 ↓
        (O - E)² / E
                 ↓
       Add all the values
                 ↓
             χ²

We need to calculate every row × every column:

                  90          110
              ┌──────────┬──────────┐
100           │ 100×90   │ 100×110  │
              │   9000   │   11000  │
              ├──────────┼──────────┤
100           │ 100×90   │ 100×110  │
              │   9000   │   11000  │
              └──────────┴──────────┘

That's exactly what np.outer() does.    



==> Then divide by the grand total:

expected = np.outer(row_totals, col_totals) / total

So:

              9000       11000
              ─────      ─────
               200        200

                ↓          ↓

                45         55

Result:

Expected:

       Bought   Didn't Buy
Male      45        55
Female    45        55

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
