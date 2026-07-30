import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    # patch * kernel
    
    """
   16 filters
   3 input channels
   Kernel height = 5
   Kernel width = 5 
    """
       # Input dimensions
    N, C_in, H, W_in = x.shape

    # Kernel dimensions
    C_out, _, KH, KW = W.shape
        # Output dimensions
    H_out = H - KH + 1
    W_out = W_in - KW + 1
        # Output tensor
    y = np.zeros((N, C_out, H_out, W_out), dtype=float)
        # Output tensor
    y = np.zeros((N, C_out, H_out, W_out), dtype=float)

    # Loop over every image in batch
    for n in range(N):

        # Loop over every filter
        for cout in range(C_out):

            # Loop over every output row
            for i in range(H_out):

                # Loop over every output column
                for j in range(W_out):

                    value = 0.0

                    # Sum over every input channel
                    for cin in range(C_in):

                        # Extract image patch
                        patch = x[n, cin, i:i+KH, j:j+KW]

                        # Corresponding kernel
                        kernel = W[cout, cin]

                        value += np.sum(patch * kernel)

                    # Add bias
                    y[n, cout, i, j] = value + b[cout]

    return y