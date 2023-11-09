import numpy as np
import matplotlib.pyplot as plt
from .initial import constructIC
from .solver import solveSWE

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
    h = [u[0] for u in U]
    u = [u[1] / u[0] for u in U]
    return h, u


def plotSWE(B, h0, u0, Nx, tEnd, timePoints, g=1, method='C', steadyH=None):
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
        steadyH     The steady height (optional)
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
        raise ValueError("Only Methods A, B, C and D have been implemented")
    if type(timePoints) != list:
        raise ValueError("timePoints should be of type list")

    # Construct the Initial Profile
    U0 = constructIC(method, h0, u0, Nx, B, g)

    # Compute the solution depending on the method
    B = np.vectorize(B)
    gridX, UHist, tHist = solveSWE(B, U0, Nx, tEnd, g, method)

    # Plot the solution at the given timestamps
    plt.figure()
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

    plt.show()

    # Plot the difference with the steady-state
    if steadyH:
        if steadyH == "TBC":
            steadyH, _ = extractVars(method, UHist[-1], g, B, gridX)
        plt.figure()
        for time in timePoints:
            if time > tHist[-1]:
                continue
            i = 0
            while tHist[i] < time: i+= 1
            h, _ = extractVars(method, UHist[i], g, B, gridX)
            plt.plot(gridX, np.array(h) - np.array(steadyH), 
                    label=f"Difference at t={round(time, 2)}")
        plt.xlim(0, 1)
        plt.title(f"Difference with Steady-State Solution for Method {method}")
        plt.xlabel(r"Spatial Coordinate $x$")
        plt.ylabel(r"Difference with Steady-State $h-\tilde h$")
        plt.legend()

        plt.show()

    # Unpack the water height and velocity from the final profile
    return extractVars(method, UHist[-1], g, B, gridX)