
def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    # output ek list chaiye
    #W is wealth factor
    #Begin with wealth factor==>
    start_wealth_factor=1.0

    #convert array into numpy array
    # returns=np.asarray(returns,dtype=float)
    res=[]
    for i in range(len(returns)):
        W_t=start_wealth_factor*(1+returns[i])
        start_wealth_factor=W_t
        R_t=W_t-1
        res.append(R_t)
    # res=np.asarray(res,dtype=float)    
    return res