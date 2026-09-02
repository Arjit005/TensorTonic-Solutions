def simple_moving_average(values: list, window_size: int) -> list:
    """
    Returns the mean of every complete sliding window.
    """
    # output should be a list
    res=[]
    for i in range(len(values)-window_size+1):
        total=0
        for j in range(i,window_size+i):
            total+=values[j]
        sma=round(total/window_size,1)
        res.append(sma)
    return res        
            
