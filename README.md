# Shallow Water Equations Solver

This package provides a well-balanced solver for the one-dimensional Saint-Venant equations, based on the principles outlined in [this paper](https://github.com/DanielCortild/SWE-Solver/blob/main/paper.pdf?raw=true) and [this presentation](https://github.com/DanielCortild/SWE-Solver/blob/main/presentation.pdf?raw=true). 

## Installation

The package is available through pip, and may be installed via:
```
pip install SWE_Solver
```

## Main Usage

To utilize this package, you can call the `plotSWE` function with the following parameters:
```
h, u = plotSWE(B, h0, u0, Nx, tEnd, timePoints, g=1, method='C')
```

### Parameters:
* **B** _(callable)_: Bottom topography function. This function defines the topographic profile and should take spatial coordinates as input and return the bottom elevation at those coordinates. 
* **h0** _(array)_: Initial water height profile. This should be an array of length `Nx`, representing the initial water height at different spatial locations.
* **u0** _(array)_: Initial water velocity profile. Similar to h0, this should be an array of length `Nx`, representing the initial water velocity at different spatial locations.
* **Nx** _(int)_: Number of spatial grid points.
* **tEnd** _(float)_: End time of the simulation. The simulation starts at time t=0.
* **timePoints** _(list)_: List of time points at which you want to visualize the results.
* **g** _(float, optional)_: Gravitational constant. Default is `1`.
* **method** _(str, optional)_: Method selection (`'A'`, `'B'` or `'C'`). Default is `'C'`.

### Returns:
* **h** _(array)_: Array containing the water height profile at the final time point.
* **u** _(array)_: Array containing the water velocity profile at the final time point.

## Pre-Coded Examples
A number of pre-coded examples are available through the library, through the function `exampleSWE`.
```
h, u = exampleSWE(state="still_flat", method='C')
```

### Parameters
* **state** _(String)_: Name of the example. Has to be one of `"still_flat"` (Constant height, zero velocity, flat bottom), `"still_tilted"` (Constant total height, zero velocity, tilted bottom),, `"still_tilted_pert"` (Perturbed constant total height, perturbed zero velocity, tilted bottom), `"moving_flat"` (Constant height, constant velocity, flat bottom), `"moving_tilted"` (Constant total height, constant velocity, tilted bottom), `"evolving_wave"` (Step function for height, constant discharge, flat bottom), `"standing_wave"` (Final profile of `"evolving_wave"` for method `'C'`, representing an equilibrium), `"standing_wave_pert"` (Final profile of `"evolving_wave"` for method `'C'`, with a perturbation), `"forming_collision"` (Constant water height, positive velocity on the right, negative velocity on the left, flat bottom), `"spike_flattening"` (Water height given by a Gaussian, zero velocity, flat bottom), `"over_bump"` (Constant total water height, constant velocity, bottom given by a Gaussian). Defaults to `"still_flat"`.
* **method** _(String)_: Name of the method used. Has to be one of `'A'`, `'B'`, `'C'`. Defaults to `'C'`.

### Returns
* **h** _(array)_: Array containing the water height profile at the final time point.
* **u** _(array)_: Array containing the water velocity profile at the final time point.

## Example

```
from SWE_Solver import plotSWE
from math import sqrt
from scipy.special import erf

Nx = 50
B = lambda x: 1
f = lambda T: 1 + sqrt(3) / (1 - erf(-0.5 / 0.1)) * (erf((T - 0.5) / 0.1) - erf (-0.5 / 0.1))
h0 = [f(_/ (Nx-1)) for _ in range(Nx)]
u0 = [2.0 / h0[_] for _ in range(Nx)]
_ = plotSWE(B, h0, u0, Nx, tEnd=1.0, timePoints=[0.0, 0.1, 0.5, 1.0])
```

The above is equivalent to the simple example given by 
```
from SWE_Solver import exampleSWE

_ = exampleSWE("evolving_wave", 'C')
```

In this example, we're using a spatial grid with 50 points, running the simulation up to `t=1` seconds, and visualizing the results at times `0.0`, `0.1`, `0.5` and `1.0` seconds, with gravitational constant `g=1` (default value) and using `method='C'` (default value).

This produces the result in the following figure.

![](https://github.com/DanielCortild/SWE-Solver/blob/main/fig.png?raw=true)