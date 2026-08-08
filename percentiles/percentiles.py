import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # concepts==>
    """
    percentiles(x, q)
      │     │
      │     └── "Which percentile?"
      │
      └──────── "From which data?"


                DATA
                  ↓
          x = [10,20,30,40,50,60]
                  ↓
          Sort the data
                  ↓
          Choose percentile
                  ↑
          q = 25, 50, 75...
                  ↓
        Find the percentile position
                  ↓
      Does position fall exactly
          on a data point?
             ↙        ↘
           YES        NO
            ↓          ↓
        Take value   Interpolate
                       between
                     two values
                         ↓
                  Final percentile
    



    Why linear?

Because we're assuming the movement between 20 and 30 is proportional:

        0%       25%             100%
        │---------│---------------│
        20       22.5            30
        
        That's the entire idea.

Linear interpolation = finding a value between two known values by moving proportionally between them.
    """
    #float() means: "Convert this thing into one Python floating-point number.for one number "

    # converting arrays into numpy array
    x=np.asarray(x,dtype=float)#  x is our data
    q=np.asarray(q,dtype=float)# it contains which  percentile we want
    # calculate percentile
    percentile=np.percentile(x,q,method='linear')

    return percentile
    
    