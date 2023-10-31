import numpy as np
import scipy as sp
from tqdm import tqdm
from .reconstruct import reconstructVars, reconstructH
from .initial import constructIC, getLambdaMax, getBLin

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
        elif method == 'D':
            w = np.array([B(x) + hi for x, hi in zip(X, h)])
            wRp, wRn, wLp, wLn = reconstructVars(w, j, dx, posPres=True)
        else:
            raise ValueError("Method is not valid")

        # Compute the Fluctuation
        fluctR = computeFluctuation(method, URp, URn, BR, F, dx, dt)
        fluctL = computeFluctuation(method, ULp, ULn, BL, F, dx, dt)

        RHS[j-1] = - (fluctR - fluctL) / dx + np.array([0, S])
        
    return RHS

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