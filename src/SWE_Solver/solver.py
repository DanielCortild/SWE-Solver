import numpy as np
import scipy as sp
from tqdm import tqdm
from .reconstruct import reconstructVars, reconstructH
from .initial import constructIC, getLambdaMax, getBLin

def computeFluctuation(Up, Un, F, dx, dt):
    """
    Computes the fluctuation at a given interfact
    Input:
        Up      Variables U at the right reconstruction of the interfact
        Un      Variables U at the left reconstruction of the interfact
        F       Flux function
        dx      Spatial cell width
        dt      Time step
    Output:
        fluct   Fluctuation at the given cell interface
    """
    return 0.5 * (F(Up) + F(Un) - dx / dt * (Up - Un))

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

    # Extracting quantities from the data
    h = [p[0] for p in U]
    q = [p[1] for p in U]
    w = [p[0] + B(x) for p, x in zip(U, X)]
    u = [p[1] / p[0] for p in U]
    E = np.array([ui ** 2 / 2 + g * (hi + B(x)) for (ui, hi, x) in zip(u, h, X)])

    # Filling up the RHS
    for j in range(1, len(RHS)+1):
        # Bottom topograhy at interfaces
        BR = B(XHalf[j])
        BL = B(XHalf[j-1])

        # Computation of Source Term
        if method == 'A':
            # Reconstructions of variables
            hRp, hRn, hLp, hLn = reconstructVars(h, j, dx)
            qRp, qRn, qLp, qLn = reconstructVars(q, j, dx)

            # Computation of source term
            S = - g * h[j] * B(X[j])
        elif method == 'B':
            # Reconstructions of variables
            wRp, wRn, wLp, wLn = reconstructVars(w, j, dx)
            qRp, qRn, qLp, qLn = reconstructVars(q, j, dx)
            hRp, hRn, hLp, hLn = wRp - B(XHalf[j]), wRn - B(XHalf[j]), wLp - B(XHalf[j-1]), wLn - B(XHalf[j-1])

            # Computation of source term
            S = - g * (w[j] - B(X[j])) * (BR - BL) / dx
        elif method == 'C':
            # Reconstructions of variables
            qRp, qRn, qLp, qLn = reconstructVars(q, j, dx)
            ERp, ERn, ELp, ELn = reconstructVars(E, j, dx)
            hRn = reconstructH(ERn, q[j], qRn, BR, h[j], g)
            hRp = reconstructH(ERp, q[j], qRp, BR, h[j], g)
            hLp = reconstructH(ELp, q[j-1], qLp, BL, h[j-1], g)
            hLn = reconstructH(ELn, q[j-1], qLn, BL, h[j-1], g)

            # Compute Source Term
            S = (- g * (hRn + hLp) / 2 * (BR - BL) / dx 
                + (hRn - hLp) / (4 * dx) * (qRn / hRn - qLp / hLp) ** 2)
        else:
            raise ValueError("Method is not valid")

        # Recontructions of U
        URp = np.array([hRp, qRp])
        URn = np.array([hRn, qRn])
        ULp = np.array([hLp, qLp])
        ULn = np.array([hLn, qLn])

        # Compute the Fluctuation
        fluctR = computeFluctuation(URp, URn, F, dx, dt)
        fluctL = computeFluctuation(ULp, ULn, F, dx, dt)

        RHS[j-1] = - (fluctR - fluctL) / dx + np.array([0, S])
        
    return RHS

def solveSWE(B, U0, Nx, t_end, g, method):
    # Create spatial grid and compute step sizes
    gridX = np.linspace(0, 1, Nx)
    dx = 1 / (Nx-1)

    # Get linearized bottom topography
    B = getBLin(B, gridX)

    # Define the Flux function
    F = lambda U: np.array([U[1], U[1] ** 2 / U[0] + g * U[0] ** 2 / 2])

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
        dt = min(dt, t_end - t)
        L = lambda U: RHS(gridX, U, B, F, dx, dt, g, method)
        Utemp1 = U0 + dt * L(U0)
        U0 = 1/2 * U0 + 1/2 * Utemp1 + 1/2 * dt * L(Utemp1)
        U_hist.append(U0.copy())
        t += dt
        t_hist.append(t)

    return gridX, U_hist, t_hist