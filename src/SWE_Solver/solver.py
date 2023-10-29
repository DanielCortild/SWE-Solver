import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm

def minmod(x):
    if min(x) >= 0:
        return min(x)
    if max(x) <= 0:
        return max(x)
    return 0

def numDer(U, j, dx):
    theta = 1
    if 1 <= j < len(U)-1:
        return minmod([theta * (U[j] - U[j-1]) / dx,
               (U[j+1] - U[j-1]) / (2 * dx),
               theta * (U[j+1] - U[j]) / dx])
    if j == 0:
        return theta * (U[j+1] - U[j]) / dx
    if j == len(U) - 1:
        return theta * (U[j] - U[j-1]) / dx

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

def reconstructH(E, q, B, g, h):
    """
    Reconstructs h given the energy and the discharge at a given point.
    Uses the previously reconstructed h as a starting point, and follows
    the algorithm laid out in the paper. 
    This is only applicable in the case of Method C
    Input:
        E           The energy at the interface
        q           The discharge at the interface
        B           The bottom topography at the interfact
        g           The gravitational constant
        h           The previously reconstructed h at the interface
    Output:
        sol         The newly reconstructed h at the interface
    """
    phi = lambda x: q ** 2 / (2 * x ** 2) + g * (x + B) - E
    Fr = abs(q) / np.sqrt(g * h ** 3)
    if q == 0:
        return E / g - B
    if Fr == 1:
        return np.cbrt(q**2 / g)
    if Fr > 1:
        hs = min(np.cbrt(q**2 / g), h)
        lamb = 0.9  
    else:
        hs = max(np.cbrt(q**2 / g), h)
        lamb = 1.1
    while phi(hs) < 1e-4: 
        hs *= lamb
    return sp.optimize.fsolve(phi, hs)[0]

def computeFluctuation(method, Up, Un, B, F, dx, dt):
    """
    Computes the fluctuation at a given interfact
    Input:
        Up      Variables U at the right reconstruction of the interfact
        Un      Variables U at the left reconstruction of the interfact
        B       Bottom topography at the interface
        F       Flux function
        dx      Spatial cell width
        dt      Time step
    Output:
        fluct   Fluctuation at the given cell interface
    """
    if method in ['A', 'C']:
        return 0.5 * (F(Up) + F(Un) - dx / dt * (Up - Un))
    elif method == 'B':
        return 0.5 * (F(Up, B) + F(Un, B) - dx / dt * (Up - Un))

def RHS(X, U, B, F, dx, dt, g, method):
    """
    Computes the RHS of a hyperbolic system, considered as a semi-discrete system
    Input:
        X           The (uniform) spatial grid
        U           The previous solution profile
        B           The bottom topogrpahy (Or its derivative)
        F           The flux function
        dx          The spatial step size
        dt          The time step size
        g           The acceleration constant
        method      The choosen method for the discretization of the source term
    Output:
        RHS         The newly evaluated RHS
    """
    # Initialise RHS and extend U and X to allow for open BCs
    RHS = np.empty_like(U)
    U = np.vstack([U[0], U, U[-1]])
    X = np.concatenate([[X[0]], X, [X[-1]]])
    XHalf = 0.5 * (X[:-1] + X[1:])

    # Filling up the RHS
    for j in range(1, len(RHS)+1):
        # Bottom topograhy at interfaces
        BR = B(XHalf[j])
        BL = B(XHalf[j-1])

        # Reconstructions of h
        h = [u[0] for u in U]
        hRp, hRn, hLp, hLn = reconstructVars(h, j, dx)

        # Reconstructions of q
        q = [u[1] for u in U]
        qRp, qRn, qLp, qLn = reconstructVars(q, j, dx)

        # Recontructions of U
        URp = np.array([hRp, qRp])
        URn = np.array([hRn, qRn])
        ULp = np.array([hLp, qLp])
        ULn = np.array([hLn, qLn])

        # Computation of Source Term
        if method == 'A':
            S = - g * h[j] * B(X[j])
        elif method == 'B':
            h_bar = h[j] - B(X[j])
            S = - g * h_bar * (BR - BL) / dx
        elif method == 'C':
            # Reconstructions of E
            E = np.array([u[1] ** 2 / (2 * u[0] ** 2) + g * (u[0] + B(X[i])) for (i, u) in enumerate(U)])
            ERp, ERn, ELp, ELn = reconstructVars(E, j, dx)

            hRn = reconstructH(ERn, qRn, BR, g, hRp)
            hRp = reconstructH(ERp, qRp, BR, g, hRn)
            hLp = reconstructH(ELp, qLp, BL, g, hLp)
            hLn = reconstructH(ELn, qLn, BL, g, hLn)

            # Compute Source Term
            S = (- g * (hRn + hLp) / 2 * (BR - BL) / dx 
                + (hRn - hLp) / (4 * dx) * (qRn / hRn - qLp / hLp) ** 2)
                # - ((qRn / hRn) ** 2 - (qLp / hLp) ** 2) / (2 * dx) * ((hRn + hLp) / 2 - 2 * hRn * hLp / (hRn + hLp)))

            # Recomputation of the reconstructions at the interfaces
            URn[0] = hRn
            URp[0] = hRp
            ULn[0] = hLn
            ULp[0] = hLp
        else:
            raise ValueError("Method is not valid")

        # Compute the Fluctuation
        fluctR = computeFluctuation(method, URp, URn, BR, F, dx, dt)
        fluctL = computeFluctuation(method, ULp, ULn, BL, F, dx, dt)

        RHS[j-1] = - (fluctR - fluctL) / dx + np.array([0, S])
        
    return RHS

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
        if x <= 0.5 * dx:
            return B(-0.5*dx) + (B(0.5*dx) - B(-0.5*dx)) * (x - (-0.5 * dx)) / (dx)
        for i in range(1, len(gridX)-1):
            hm = 0.5 * (gridX[i-1] + gridX[i])
            hp = 0.5 * (gridX[i] + gridX[i+1])
            if  hm <= x <= hp:
                return B(hm) + (B(hp) - B(hm)) * (x - hm) / (hp - hm)
        if x >= 1 - 0.5 * dx:
            return B(1-0.5*dx) + (B(1+0.5*dx) - B(1-0.5*dx)) * (x - (1-0.5*dx)) / (dx)
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
    return max([U[j][1]/U[j][0] + np.sqrt(max(g * U[j][0], 0)) for j in range(len(U))])

def solveSWE(B, U0, Nx, t_end, g, method):
    # Create spatial grid and compute step sizes
    gridX = np.linspace(0, 1, Nx)
    dx = 1 / (Nx-1)

    # Get linearized bottom topography
    B = getBLin(B, gridX)

    # Define the Flux function
    if method in ['A', 'C']:
        F = lambda U: np.array([U[1], U[1] ** 2 / U[0] + g * U[0] ** 2 / 2])
    elif method == 'B':
        F = lambda U, x: np.array([U[1], U[1] ** 2 / (U[0] - B(x)) + g * (U[0] - B(x)) ** 2 / 2])

    # Solve the system using SSPRK(2,2)
    U_hist = [U0.copy()]
    t_hist = [0]
    t = 0
    
    def generator(t_end):
        while t < t_end:
            yield

    pbar = tqdm(generator(t_end))
    dt = min(dx / getLambdaMax(U0, g), t_end)
    for _ in pbar:
        pbar.set_description(f"Total Time: {round(t, 4)} / {t_end}")
        # print(dt)
        dt = min(dt, t_end - t)
        L = lambda U: RHS(gridX, U, B, F, dx, dt, g, method)
        Utemp1 = U0 + dt * L(U0)
        U0 = 1/2 * U0 + 1/2 * Utemp1 + 1/2 * dt * L(Utemp1)
        # U0 += dt * L(U0)
        U_hist.append(U0.copy())
        t += dt
        t_hist.append(t)

    return gridX, U_hist, t_hist

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

    # Dependeing on the method, construct the variables
    if method in ['A', 'C']:
        # In Methods A and C, the variables are height (h) and discharge (q=hu)
        U0 = np.array([[h, u*h] for h, u in zip(h0, u0)])
    elif method == 'B':
        # In Method B, the variables are total height (w=h+B) and discharge (q=hu)
        U0 = np.array([[B(x) + h, u*h] for h, u, x in zip(h0, u0, X)])
    else:
        raise ValueError(f"Intial Condition for Method {method} not implemented")
    return U0

def extractVars(method, U, g, B, gridX):
    """
    Extract the water height h and velocity u from a given profile
    Input:
        method      Which method has been used to couple the variables
        U           The profile of variables
        g           The graviational constant
        B           The bottom topography function
        gridX       The spatial grid
    Output:
        h           The water height profile
        u           The water velocity profile
    """
    if method in ['A', 'C']:
        h = [u[0] for u in U]
        u = [u[1] / u[0] for u in U]
    elif method == 'B':
        h = [u[0] - B(x) for u, x in zip(U, gridX)]
        u = [u[1] / (u[0] - B(x)) for u, x in zip(U, gridX)]
    else:
        raise ValueError(f"Extracting h has not been implemented for method {method}")
    return h, u


def plotSWE(B, h0, u0, Nx, tEnd, timePoints, g=1, method='C'):
    """
    Computes and plots the solution to the SWE equations.
    Input:
        B           Bottom topography (function)
        h0          Initial water height profile
        u0          Initial water velocity profile
        Nx          Number of spatial gridpoints
        tEnd        End time of simulation
        timePoints  Points in time to include in plot
        g           Gravitational constant
        method      Method used for discretization
    Output:
        h           Final water height profile
        u           Final water velocity profile
    """
    # Check if variables make sense
    if not callable(B) and (type(B) == list and (not callable(B[0]) or not callable(B[1]))):
        raise ValueError("Bottom topography B should be of type callable")
    if type(h0) != list or type(u0) != list:
        raise ValueError("h0 and u0 should be of type list")
    if len(h0) != len(u0):
        raise ValueError("h0 and u0 should have same length")
    if Nx <= 1:
        raise ValueError("Nx must be at least 2")
    if len(h0) != Nx:
        raise ValueError("h0 and u0 should have length Nx")
    if tEnd <= 0:
        raise ValueError("tEnd should be larger than 0")
    if g <= 0:
        raise ValueError("g should be positive")
    if method not in ['A', 'B', 'C']:
        raise ValueError("Only Methods A, B and C have been implemented")
    if type(timePoints) != list:
        raise ValueError("timePoints should be of type list")

    # Construct the Initial Profile
    U0 = constructIC(method, h0, u0, Nx, B, g)

    # Compute the solution depending on the method
    if method == 'A':
        B, Bx = B
        B = np.vectorize(B)
        Bx = np.vectorize(Bx)
        gridX, UHist, tHist = solveSWE(Bx, U0, Nx, tEnd, g, method)
    elif method in ['B', 'C']:
        B = np.vectorize(B)
        gridX, UHist, tHist = solveSWE(B, U0, Nx, tEnd, g, method)
    else:
        raise ValueError(f"Method {method} not implemented")

    # Plot the solution at the given timestamps
    plt.fill_between(gridX, B(gridX), label="Bottom Topography", color="brown")
    for time in timePoints:
        if time > tHist[-1]:
            continue
        i = 0
        while tHist[i] < time: i+= 1
        h, _ = extractVars(method, UHist[i], g, B, gridX)
        plt.plot(gridX, B(gridX) + h, 
                label=f"Water Height at t={round(time, 2)}")
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.title(f"Shallow-Water Profile for Method {method}")
    plt.xlabel(r"Spatial Coordinate $x$")
    plt.ylabel(r"Water depth $B+h$")
    plt.legend()

    # Unpack the water height and velocity from the final profile
    return extractVars(method, UHist[-1], g, B, gridX)