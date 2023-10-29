# Shallow Water Equations Solver

This package provides a well-balanced solver for the one-dimensional Saint-Venant equations, based on the principles outlined in [this paper](XXX).

## Usage

To utilize this package, you can call the `SWE_Plot`` function with the following parameters:

```
h, u = SWE_Plot(B, h0, u0, Nx, tEnd, times, g=1, method='C')
```
### Parameters:
* **B** _(callable)_: Bottom topography function. This function defines the topographic profile and should take spatial coordinates as input and return the bottom elevation at those coordinates. In case of method 'A', should be an array of two callables, representing B and its derivative.
* **h0** _(array)_: Initial water height profile. This should be an array of length Nx, representing the initial water height at different spatial locations.
* **u0** _(array)_: Initial water velocity profile. Similar to h0, this should be an array of length Nx, representing the initial water velocity at different spatial locations.
* **Nx** _(int)_: Number of spatial grid points.
* **tEnd** _(float)_: End time of the simulation. The simulation starts at time t=0.
* **times** _(list)_: List of time points at which you want to visualize the results.
* **g** _(float, optional)_: Gravitational constant. Default is 1.
* **method** _(str, optional)_: Method selection ('A', 'B' or 'C'). Default is 'C'.

### Returns:
* **h** _(array)_: Array containing the water height profile at the final time point.
* **u** _(array)_: Array containing the water velocity profile at the final time point.

## Example:

```
B = lambda x: 1
f = lambda T: 1 + sqrt(3) / (1 - erf(-0.5 / 0.1)) * (erf((T - 0.5) / 0.1) - erf (-0.5 / 0.1))
h0 = [f(_/ (Nx-1)) for _ in range(Nx)]
u0 = [2.0 / h0[_] for _ in range(Nx)]
_ = SWE_Plot(B, h0, u0, Nx=100, tEnd=1.0, times=[0.0, 0.1, 0.5, 1.0])
```

In this example, we're using a spatial grid with 100 points, running the simulation up to `t=1` seconds, with gravitational constant `g=1` (default value), and visualizing the results at times `0.0`, `0.1`, `0.5` and `1.0` seconds using method 'C'.

This produces the result in the following figure.

<img src="./fig.png" style="display: block; margin-left: auto; margin-right: auto;"/>