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
    # Parameter, could be tweaked
    theta = 1

    # General case
    if 1 <= j < len(U)-1:
        return minmod([theta * (U[j] - U[j-1]) / dx,
               (U[j+1] - U[j-1]) / (2 * dx),
               theta * (U[j+1] - U[j]) / dx])

    # Left-boundary case
    if j == 0:
        return theta * (U[j+1] - U[j]) / dx

    # Right-boundary case
    if j == len(U) - 1:
        return theta * (U[j] - U[j-1]) / dx

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
    # Pathological case 
    if h < 1e-8: return h

    # Function of which we seek a minimum
    phi = lambda x: qHalf ** 2 / (2 * x ** 2) + g * (x + BHalf) - EHalf

    # Compute sonic point
    h0 = np.cbrt(qHalf ** 2 / g)

    # Compute froude approximation
    Fr = abs(q) / np.sqrt(g * h ** 3)

    # Treat trivial cases
    if abs(q) < 1e-8: return EHalf / g - BHalf
    if abs(Fr - 1) < 1e-4: return h0

    # Set initial guess
    hStar, lamb = [min(h, h0), 0.9] if Fr > 1 else [max(h, h0), 1.1]
    while phi(hStar) < 1e-4: hStar *= lamb

    # Solve using Newton's method, and in case of non-convergence return h0
    try:
        return sp.optimize.newton(phi, hStar)
    except:
        return h0