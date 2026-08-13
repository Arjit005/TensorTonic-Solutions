import numpy as np


def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h  (shape: H,)
        dW: gradient wrt W                (shape: H x D)
        dU: gradient wrt U                (shape: H x H)
        db: gradient wrt bias             (shape: H,)
    """

    # cache is not a mathematical variable
    # This is a huge source of confusion.
    # cache is just a container.
    
    x_t, h_prev, h_t, W, U, b = cache

    # converting input values from Python lists
    # into NumPy arrays so vector operations work
    dh = np.asarray(dh)
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    h_t = np.asarray(h_t)
    W = np.asarray(W)
    U = np.asarray(U)
    b = np.asarray(b)

    """
                     FORWARD

       x_t ──→ W·x_t ──┐
                       │
       h_prev → U·h ───┼──→ + b → tanh → h_t
                       │
                       ▼


                    BACKWARD

                       dh
                        │
                        ▼
                tanh derivative
                        │
                        ▼
                da = dh * (1 - h_t²)
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
           W.T        U.T           1
             │          │           │
             ▼          ▼           ▼
            dx        dh_prev      db

             │           │
             ▼           ▼
         da · x_tᵀ   da · h_prevᵀ
             │           │
             ▼           ▼
            dW          dU


                  🔄 FULL CYCLE

        x_t + h_prev + W + U + b
                    ↓
                  h_t
                    ↓
                   dh
                    ↓
                  da
                    ↓
          ┌─────────┼─────────┐
          ↓         ↓         ↓
         dx       dh_prev     db
          ↓
         dW / dU
    """

    da = dh * (1 - h_t ** 2)

    dx_t = W.T @ da

    dh_prev = U.T @ da

    dW = np.outer(da, x_t)

    dU = np.outer(da, h_prev)

    db = da

    return dx_t, dh_prev, dW, dU, db