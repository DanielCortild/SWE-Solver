from .post_processing import plotSWE
import numpy as np
from scipy.special import erf

# Nx = 50
# tEnd = 0.1
# times = [0.0, 0.05, 0.1]

def exampleSWE(state='still_flat', method='C'):
    """
    Helper function to run pre-coded examples
    Input:
        state       The example to run
        method      The method used to run the example
    Output:
        h           The final water height profile
        u           The final water velocity profile
    """
    g = 1

    if state == "still_flat":
        Nx = 50
        B = lambda x: 1
        Bx = lambda x: 0
        h0 = [4.0 - B(_ / (Nx-1)) for _ in range(Nx)]
        u0 = [0.0 for _ in range(Nx)]
        tEnd = 1.0
        timePoints = [0.0, 0.5, 1.0]
    elif state == "still_tilted":
        Nx = 50
        B = lambda x: x
        Bx = lambda x: 1
        h0 = [4.0 - B(_ / (Nx-1)) for _ in range(Nx)]
        u0 = [0.0 for _ in range(Nx)]
        tEnd = 1.0
        timePoints = [0.0, 0.5, 1.0]
    elif state == "moving_flat":
        Nx = 50
        B = lambda x: 1
        Bx = lambda x: 0
        h0 = [4.0 - B(_ / (Nx-1)) for _ in range(Nx)]
        u0 = [1.0 for _ in range(Nx)]
        tEnd = 1.0
        timePoints = [0.0, 0.5, 1.0]
    elif state == "moving_tilted":
        Nx = 50
        B = lambda x: x
        Bx = lambda x: 1
        h0 = [4.0 - B(_ / (Nx-1)) for _ in range(Nx)]
        u0 = [1.0 for _ in range(Nx)]
        tEnd = 1.0
        timePoints = [0.0, 0.5, 1.0]
    elif state == "evolving_wave":
        Nx = 50
        B = lambda x: 1
        Bx = lambda x: 0
        Amin = 1
        Amax = 1 + sqrt(3)
        T0 = 0.5
        eps = 0.1
        f = lambda T: Amin + (Amax - Amin) / (1 - erf(-T0 / eps)) * (erf((T - T0) / eps) - erf (-T0 / eps))
        h0 = [f(_/ (Nx-1)) for _ in range(Nx)]
        u0 = [2.0 / h0[_] for _ in range(Nx)]
        tEnd = 1.0
        timePoints = [0.0, 0.1, 0.5, 1.0]
    elif state == "forming_collision":
        Nx = 50
        B = lambda x: 1
        h0 = [4 for _ in range(Nx)]
        def u(x):
            if 0 <= x <= 0.3:
                return 1
            if 0.3 < x < 0.7:
                return 0
            if 0.7 <= x <= 1:
                return -1
        u0 = [u(_ / (Nx - 1)) for _ in range(Nx)]
        tEnd = 0.5
        timePoints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    elif state == "spike_flattening":
        Nx = 50
        B = lambda x: 1
        f = lambda x: np.exp(- (x - 0.5) ** 2 / (2 * 0.01))
        h0 = [2.0 + f(_ / (Nx - 1)) for _ in range(Nx)]
        u0 = [0 for _ in range(Nx)]
        tEnd = 1.0
        timePoints = [0.0, 0.1, 0.5, 1.0]
    elif state == "over_bump":
        Nx = 50
        B = lambda x: np.exp(- (x - 0.5) ** 2 / (2 * 0.01))
        h0 = [4.0 - B(_ / (Nx - 1)) for _ in range(Nx)]
        u0 = [1 for _ in range(Nx)]
        tEnd = 1.0
        timePoints = [0.0, 0.1, 0.5, 1.0]
    elif state == "half_dry":
        B = lambda x: 1
        h0 = [4 if _ <= Nx / 2 else 0 for _ in range(Nx)]
        u0 = [0 for _ in range(Nx)]
    elif state == "hitting_box":
        B = lambda x: 1 if 0.7 <= x <= 0.9 else 0.2
        h0 = [4 if _ <= Nx / 2 else 0.1 for _ in range(Nx)]
        u0 = [0 for _ in range(Nx)]
    else:
        raise ValueError("Example state not implemented")

    if method in ['B', 'C']:
        return plotSWE(B, h0, u0, Nx, tEnd, timePoints, g, method)
    elif method == 'A':
        return plotSWE([B, Bx], h0, u0, Nx, tEnd, timePoints, g, method)
    else:
        raise ValueError("Method not implemented")