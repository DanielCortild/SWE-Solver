import numpy as np

def getBLin(B, gridX):
    """
    Computes a linear approximation of a function
    Input:
        B           The function to be approximated
        gridX       The spatial grid
    Output:
        BLin        The linearized function
    """
    dx = gridX[1]
    def BLin(x):
        if -0.5 * dx <= x <= 0:
            return B(0.5*dx) + (B(-0.5*dx) - B(0.5*dx)) * (x - (-0.5 * dx)) / (dx)
        if 0 <= x <= 0.5 * dx:
            return B(-0.5*dx) + (B(0.5*dx) - B(-0.5*dx)) * (x - (-0.5 * dx)) / (dx)
        for i in range(1, len(gridX)-1):
            hm = 0.5 * (gridX[i-1] + gridX[i])
            hp = 0.5 * (gridX[i] + gridX[i+1])
            if  hm <= x <= hp:
                return B(hm) + (B(hp) - B(hm)) * (x - hm) / (hp - hm)
        if 1 - 0.5 * dx <= x <= 0:
            return B(1-0.5*dx) + (B(1+0.5*dx) - B(1-0.5*dx)) * (x - (1-0.5*dx)) / (dx)
        if 0 <= x <= 1 + 0.5 * dx:
            return B(1+0.5*dx) + (B(1-0.5*dx) - B(1+0.5*dx)) * (x - (1-0.5*dx)) / (dx)
        raise ValueError("Cannot evaluate BLin at that point")
    return np.vectorize(BLin)

def getLambdaMax(U, g):
    """
    Computes the maximal eigenvalue of the flux function in the SWE equations.
    The eigenvalue is computed as the maximal value taken by u+sqrt(gh), over 
    all u and h.
    Input:
        U       The variable vector, containing water height (h) and discharge (q=hu)
        g       The gravitational constant
    Output:
        lamb    The maximal eigenvalue
    """
    return max([u[1]/u[0] + np.sqrt(max(g * u[0], 0)) for u in U])


def constructIC(method, h0, u0, Nx, B, g):
    """
    Constructs the initial condition given the vectors h0 an u0.
    Depending on the method, the variables are not h and u, and hence a variable transformation is required.
    Input:
        method      The method used
        h0          The initial profile in water height
        u0          The initial profile in water velocity
        Nx          The number of gridpoints
        B           The bottom topography function
        g           The gravitational constant
    """
    # Construct and extend the grid
    X = np.linspace(0, 1, Nx)

    # Construct the variables
    U0 = np.array([[h, u*h] for h, u in zip(h0, u0)])
    return U0