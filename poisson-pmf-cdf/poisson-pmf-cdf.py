import math
import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    k_factorial=math.factorial(k)
    poisson_pmf=np.exp(-lam)*(lam**k)/k_factorial

    poisson_cdf=0
    for i in range(k+1):
        poisson_cdf+=(np.exp(-lam)*(lam**i)/math.factorial(i))
    return float(poisson_pmf),float(poisson_cdf)    