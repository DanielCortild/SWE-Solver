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

def reconstructVars(vars, dx):
    """
    Reconstructs variables according to the linear derivative approximation.
    Input:
        vars        List of variables at grid points
        dx          Spatial discretization cell width
    Output
        varsn       Reconstructed Variable at right of interfaces
        varsp       Reconstructed Variable at left of interfaces
    """
    varsn = [vL + dx * numDer(vars, j, dx) / 2 for (j, vL) in enumerate(vars)][:-1]
    varsp = [vR - dx * numDer(vars, j, dx) / 2 for (j, vR) in enumerate(vars)][1:]
    return varsn, varsp

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
    Fr = abs(q) / np.sqrt(g * h ** 3) #abs(q) / np.sqrt(max(g * h ** 3, 1e-4))
    h0 = np.cbrt(qHalf ** 2 / g)
    if q == 0: return EHalf / g - BHalf
    if Fr == 1: return h0
    hStar, lamb = [min(h0, h), 0.9] if Fr > 1 else [max(h0, h), 1.1]
    while phi(hStar) < 1e-4: 
        hStar *= lamb
    try:
        return sp.optimize.newton(phi, hStar)
    except:
        return h0
        if abs(Fr - 1) > 0.2:
            print(f"Newton did not converge and Fr={Fr}")