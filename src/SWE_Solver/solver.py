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
    F = lambda h, q: np.array([q, q ** 2 / h + g * h ** 2 / 2])
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
    U = np.vstack([U[0], U, U[-1]])
    X = np.concatenate([[X[0]], X, [X[-1]]])
    XHalf = 0.5 * (X[:-1] + X[1:])
    BX = B(X)
    BHalf = B(XHalf)

    # Extracting quantities from the data
    q = [p[1] for p in U]
    u = [p[1] / p[0] for p in U]
    h = [p[0] for p in U]

    # Reconstruct q (Same for all methods)
    qn, qp = reconstructVars(q, dx)
    
    # Reconstruct h (Different per method)
    if method == 'A':
        # For method A, we directly reconstruct h
        hn, hp = reconstructVars(h, dx)
    if method == 'B':
        # For method B, reconstruction of h is based on reconstruction of w
        w = [p[0] + B(x) for p, x in zip(U, X)]
        wn, wp = reconstructVars(w, dx)

        hn = [w_n - B_ for w_n, B_ in zip(wn, BHalf)]
        hp = [w_p - B_ for w_p, B_ in zip(wp, BHalf)]
    if method == 'C':
        # For method C, reconstruction of h is based on reconstruction of E
        E = np.array([u_ ** 2 / 2 + g * (h_ + B_) for (u_, h_, B_) in zip(u, h, BX)])
        En, Ep = reconstructVars(E, dx)

        hn = [reconstructH(E_n, qL, q_n, B_, hL, g) for E_n, qL, q_n, B_, hL in zip(En, q[:-1], qn, BHalf, h[:-1])]
        hp = [reconstructH(E_p, qR, q_p, B_, hL, g) for E_p, qR, q_p, B_, hL in zip(Ep, q[1:], qp, BHalf, h[:-1])]

    Up = np.stack([hp, qp], axis=1)
    Un = np.stack([hn, qn], axis=1)

    # Computing the RHS
    for j in range(1, len(U)-1):
        # Bottom topograhy at interfaces
        BR, BL = B(XHalf[j]), B(XHalf[j-1])

        # Computation of Source Term
        if method == 'A':
            S = - g * h[j] * B(X[j])
        elif method == 'B':
            S = - g * (w[j] - B(X[j])) * (BR - BL) / dx
        elif method == 'C':
            qRn, qLp = qn[j], qp[j-1]
            hRn, hLp = hn[j], hp[j-1]
            S = (- g * (hRn + hLp) / 2 * (BR - BL) / dx + (hRn - hLp) / (4 * dx) * (qRn / hRn - qLp / hLp) ** 2)
        else:
            raise ValueError("Method is not valid")

        # Compute the Fluctuation
        fluctR = computeFluctuation(Up[j], Un[j], dx, dt, g)
        fluctL = computeFluctuation(Up[j-1], Un[j-1], dx, dt, g)

        RHS[j-1] = - (fluctR - fluctL) / dx + np.array([0, S])

    # print(RHS)

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
    dt = min(dx / getLambdaMax(U0, g), t_end)
    for _ in pbar:
        pbar.set_description(f"Total Time: {round(t, 4)} / {t_end}")
        dt = min(dt, t_end - t)
        L = lambda U: RHS(gridX, U, B, dx, dt, g, method)
        Utemp1 = U0 + dt * L(U0)
        U0 = 1/2 * U0 + 1/2 * Utemp1 + 1/2 * dt * L(Utemp1)
        U_hist.append(U0.copy())
        t += dt
        t_hist.append(t)

    return gridX, U_hist, t_hist