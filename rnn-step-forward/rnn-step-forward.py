import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    """
    The core RNN equation you're implementing is:
    
                     ┌─────────────┐
    x_t ──→ Wx ─────→│             │
                     │     + b     │──→ tanh ──→ h_t
    h_prev → Wh ────→│             │
                     └─────────────┘
    """
    # step 1: converting input into np arrays
    x_t=np.asarray(x_t,dtype=float)
    h_prev=np.asarray(h_prev,dtype=float)
    
    #hidden state update
    pre_act = x_t @ Wx + h_prev @ Wh + b
    h_t=np.tanh(pre_act)

    return h_t
    
    
    
