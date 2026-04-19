import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x=np.array(x,dtype='float')
    mean=p
    var=p*(1-p)
    pmf=np.where(x==1,p,1-p)
    # for x in x:
    #     if x==1:
    #         pmf.append(p)
    #     else:
    #         pmf.append(1-p)
        
    return (pmf,mean,var)
    