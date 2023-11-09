import numpy as np
import scipy as sp
from tqdm import tqdm
from .reconstruct import reconstructVars, reconstructH, numDer
from .initial import constructIC, getLambdaMax, getBLin

def computeFluctuation(Up, Un, dx, dt, g):
    """
    Computes the fluctuation at a given interfact
    Input:
        Up      Variables U at the right reconstruction of the interfact
        Un      Variables U at the left reconstruction of the interfact
        dx      Spatial cell width
        dt      Time step
    Output:
        fluct   Fluctuation at the given cell interface
    """
    def F (h, q): 
        u = q / h if h >= 1e-8 else 0
        return np.array([h * u, h * u ** 2 + g * h ** 2 / 2])
    return 0.5 * (F(*Up) + F(*Un) - dx / dt * (Up - Un))

def RHS(X, U, B, dx, dt, g, method):
    """
    Computes the RHS of a hyperbolic system, considered as a semi-discrete system
    Input:
        X           The (uniform) spatial grid
        U           The previous solution profile
        B           The bottom topogrpahy (Or its derivative)
        dx          The spatial step size
        dt          The time step size
        g           The acceleration constant
        method      The choosen method for the discretization of the source term
    Output:
        RHS         The newly evaluated RHS
    """
    # Initialise RHS before changing the variables
    RHS = np.empty_like(U)

    # Extend U and X to allow for open BCs
    U = np.vstack([U[1], U, U[-2]])
    X = np.concatenate([[X[1]], X, [X[-2]]])
    XHalf = 0.5 * (X[:-1] + X[1:])
    BX = B(X)
    BHalf = B(XHalf)

    # Extracting quantities from the data
    u = [p[1] / p[0] if p[0] > 1e-4 else 0 for p in U]
    h = [p[0] for p in U]
    q = [h_ * u_ for h_, u_ in zip(h, u)]

    # Reconstruct q (Same for all methods)
    qn, qp = reconstructVars(q, dx)
    
    # Reconstruct h (Different per method)
    if method == 'A':
        # For method A, we directly reconstruct h
        hn, hp = reconstructVars(h, dx)
    if method == 'B':
        # For method B, reconstruction of h is based on reconstruction of w
        w = [p[0] + Bx for p, Bx in zip(U, BX)]
        wn, wp = reconstructVars(w, dx)

        hn = [w_n - B_ for w_n, B_ in zip(wn, BHalf)]
        hp = [w_p - B_ for w_p, B_ in zip(wp, BHalf)]
    if method == 'C':
        # For method C, reconstruction of h is based on reconstruction of E
        E = np.array([u_ ** 2 / 2 + g * (h_ + B_) for (u_, h_, B_) in zip(u, h, BX)])
        En, Ep = reconstructVars(E, dx)

        hn = [reconstructH(E_n, qL, q_n, B_, hL, g) for E_n, qL, q_n, B_, hL, hR in zip(En, q[:-1], qn, BHalf, h[:-1], h[1:])]
        hp = [reconstructH(E_p, qR, q_p, B_, hL, g) for E_p, qR, q_p, B_, hL, hR in zip(Ep, q[1:], qp, BHalf, h[:-1], h[1:])]

        un = [q_n / h_n if h_n > 1e-4 else 0 for q_n, h_n in zip(qn, hn)]
        up = [q_p / h_p if h_p > 1e-4 else 0 for q_p, h_p in zip(qp, hp)]

        qn = [u_n * h_n for u_n, h_n in zip(un, hn)]
        qp = [u_p * h_p for u_p, h_p in zip(up, hp)]

    Up = np.stack([hp, qp], axis=1)
    Un = np.stack([hn, qn], axis=1)

    # Computing the fluctuations
    fluctR = np.empty((len(U)-2, 2))
    fluctL = np.empty((len(U)-2, 2))
    sourceTerms = np.empty(len(U)-2)
    for j in range(1, len(U)-1):
        # Bottom topograhy at interfaces
        BR, BL = BHalf[j], BHalf[j-1]

        # Computation of Source Term
        if method == 'A' or method == 'B':
            S = - g * h[j] * (BR - BL) / dx
        elif method == 'C':
            hRn, hLp = hn[j], hp[j-1]
            uRn, uLp = un[j], up[j-1]
            S = (- g * (hRn + hLp) / 2 * (BR - BL) / dx + (hRn - hLp) / (4 * dx) * (uRn - uLp) ** 2)
        else:
            raise ValueError("Method is not valid")

        # Compute the Fluctuation
        fluctR[j-1] = computeFluctuation(Up[j], Un[j], dx, dt, g)
        fluctL[j-1] = computeFluctuation(Up[j-1], Un[j-1], dx, dt, g)

        sourceTerms[j-1] = S
        
    RHS = - dt * (fluctR - fluctL) / dx + dt * np.stack([np.zeros_like(sourceTerms), sourceTerms], axis=1)

    return RHS

def solveSWE(B, U0, Nx, t_end, g, method):
    # Create spatial grid and compute step sizes
    gridX = np.linspace(0, 1, Nx)
    dx = 1 / (Nx-1)

    # Get linearized bottom topography
    B = getBLin(B, gridX)

    # Solve the system using SSPRK(2,2)
    U_hist = [U0.copy()]
    t_hist = [0]
    t = 0
    
    def generator(t_end):
        while t < t_end:
            yield

    pbar = tqdm(generator(t_end))
    dt = min(dx / (2 * getLambdaMax(U0, g)), t_end - t)
    for _ in pbar:
        pbar.set_description(f"Total Time: {round(t, 4)} / {t_end}")
        L = lambda U: RHS(gridX, U, B, dx, dt, g, method)
        Utemp1 = U0 + L(U0)
        U0 = 1/2 * U0 + 1/2 * Utemp1 + 1/2 * L(Utemp1)
        U_hist.append(U0.copy())
        t += dt
        t_hist.append(t)

    return gridX, U_hist, t_hist