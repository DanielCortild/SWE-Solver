import numpy as np
import scipy as sp

def minmod(x):
    """
    Implements the minmod function on an array
    Input:
        x       Array of values
    Output:
        mx      The minmod of the array (scalar)
    """
    if min(x) >= 0:
        return min(x)
    if max(x) <= 0:
        return max(x)
    return 0

def numDer(U, j, dx):
    """
    Returns the numerical derivative of an element in an array
    Input:
        U           The array of elements
        j           The index at which the numerical derivative is to be taken
        dx          The spatial step size
    Output:
        numder      The numerical derivative at index j
    """
    theta = 1
    if 1 <= j < len(U)-1:
        return minmod([theta * (U[j] - U[j-1]) / dx,
               (U[j+1] - U[j-1]) / (2 * dx),
               theta * (U[j+1] - U[j]) / dx])
    if j == 0:
        return theta * (U[j+1] - U[j]) / dx
    if j == len(U) - 1:
        return theta * (U[j] - U[j-1]) / dx
    return 0

def reconstructVars(vars, j, dx):
    """
    Reconstructs variables according to the linear derivative approximation.
    Input:
        vars        List of variables at grid points
        j           Index j for which we which to reconstruct vRp, vRn, vLp, vLn
        dx          Spatial discretization cell width
    Output
        vRp         Reconstructed Variable at Right interface, from the right
        vRn         Reconstructed Variable at Right interface, from the left
        vLp         Reconstructed Variable at Left interface, from the right
        vLn         Reconstructed Variable at Left interface, from the left
    """
    vRp = vars[j+1] - dx * numDer(vars, j+1, dx) / 2
    vRn = vars[j] + dx * numDer(vars, j, dx) / 2
    vLp = vars[j] - dx * numDer(vars, j, dx) / 2
    vLn = vars[j-1] + dx * numDer(vars, j-1, dx) / 2
    return vRp, vRn, vLp, vLn

def reconstructH(EHalf, q, qHalf, BHalf, h, g):
    """
    Reconstructs h given the energy and the discharge at a given point.
    Uses the previously reconstructed h as a starting point, and follows
    the algorithm laid out in the paper. 
    This is only applicable in the case of Method C
    Input:
        EHalf       The energy at the interface
        q           The discharge at the center
        qHalf       The discharge at the interface
        BHalf       The bottom topography at the interfact
        h           The water height at the center
        g           The gravitational constant
    Output:
        sol         The newly reconstructed h at the interface
    """
    phi = lambda x: qHalf ** 2 / (2 * x ** 2) + g * (x + BHalf) - EHalf
    Fr = abs(q) / np.sqrt(g * h ** 3)
    h0 = np.cbrt(qHalf ** 2 / g)
    if q == 0:
        return EHalf / g - BHalf
    if Fr == 1:
        return h0
    if Fr > 1:
        hs = min(h0, h)
        lamb = 0.9  
    else:
        hs = max(h0, h)
        lamb = 1.1
    while phi(hs) < 1e-4: 
        hs *= lamb
    return sp.optimize.fsolve(phi, hs)[0]