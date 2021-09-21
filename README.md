[![PyPi Version](https://img.shields.io/pypi/v/cleanode.svg?style=flat-square)](https://pypi.org/project/cleanode)

# CleanODE
Customized collection of ODE solvers

____
## Installation:
```
pip install cleanode
```
____

## List of embedded solvers:

### Explicit:
EulerODESolver

MidpointODESolver

RungeKutta4ODESolver

Fehlberg45Solver

Ralston2ODESolver

RungeKutta3ODESolver

Heun3ODESolver

Ralston3ODESolver

SSPRK3ODESolver

Ralston4ODESolver

Rule384ODESolver

HeunEuler21ODESolver

Fehlberg21ODESolver

BogackiShampine32ODESolver

CashKarp54ODESolver

DormandPrince54ODESolver

### Implicit:

EverhartIIRadau7ODESolver

EverhartIIRadau15ODESolver

EverhartIIRadau21ODESolver

EverhartIILobatto7ODESolver

EverhartIILobatto15ODESolver

EverhartIILobatto21ODESolver

*to be continued...* 

____
## Example:

```python
import math

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import scipy.constants as const
from cleanode.ode_solvers import *


# Example of the system ODE solving: simple orbit
if __name__ == '__main__':
    # noinspection PyUnusedLocal
    def f(u: List[float], t: Union[np.ndarray, np.float64]) -> List:
        """
        Calculation of the right parts of the ODE system
        :param u: values of variables
        :type u: List[float]
        :param t: time
        :type t: Union[np.ndarray, np.float64]
        :return: calculated values of the right parts
        :rtype: np.ndarray
        """

        # Mathematically, the ODE system looks like this:
        # dx/dt = Vx
        # dVx/dt = -x / sqrt(x^2 + y^2 + z^2)^3
        # dy/dt = Vy
        # dVy/dt = -y / sqrt(x^2 + y^2 + z^2)^3
        # dz/dt = Vz
        # dVz/dt = -z / sqrt(x^2 + y^2 + z^2)^3

        g = const.g

        x = u[0]
        vx = u[1]
        y = u[2]
        vy = u[3]
        z = u[4]
        vz = u[5]

        right_sides = [
            vx,
            -x / math.sqrt(x**2 + y**2 + z**2)**3,
            vy,
            -y / math.sqrt(x**2 + y**2 + z**2)**3,
            vz,
            -z / math.sqrt(x**2 + y**2 + z**2)**3
            ]

        return right_sides

    # noinspection PyUnusedLocal
    def f2(u: np.longdouble, du_dt: np.longdouble, t: Union[np.ndarray, np.longdouble]) -> np.array:
        """
        Calculating the right side of the 2nd order ODE
        :param u: variable
        :type u: np.longdouble
        :param du_dt: time derivative of variable
        :type du_dt: np.longdouble
        :param t: time
        :type t: Union[np.ndarray, np.float64]
        :return: calculated value of the right part
        :rtype: np.array
        """

        # Mathematically, the ODE system looks like this:
        # d(dx)/dt^2 = -x / sqrt(x^2 + y^2 + z^2)^3
        # d(dy)/dt^2 = -y / sqrt(x^2 + y^2 + z^2)^3
        # d(dz)/dt^2 = -z / sqrt(x^2 + y^2 + z^2)^3

        x = u[0]
        y = u[1]
        z = u[2]

        right_sides = np.array([
            -x / math.sqrt(x**2 + y**2 + z**2)**3,
            -y / math.sqrt(x**2 + y**2 + z**2)**3,
            -z / math.sqrt(x**2 + y**2 + z**2)**3
        ], dtype='longdouble')

        return right_sides


    def exact_f(t):
        x = np.sin(t)
        y = np.cos(t)
        return x, y

    # calculation parameters:
    t0 = np.longdouble(0)
    tmax = np.longdouble(10 * math.pi)
    dt0 = np.longdouble(0.01)

    tolerance = 1e-4

    # initial conditions:
    x0 = np.longdouble(0)
    y0 = np.longdouble(1)
    z0 = np.longdouble(0)
    vx0 = np.longdouble(1)
    vy0 = np.longdouble(0)
    vz0 = np.longdouble(0)

    u0 = np.array([x0, vx0, y0, vy0, z0, vz0], dtype='longdouble')
    solver = Fehlberg45Solver(f, u0, t0, tmax, dt0, is_adaptive_step=True, tolerance=tolerance)
    solution, time_points = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    x_solution = solution[:, 0]
    y_solution = solution[:, 2]
    z_solution = solution[:, 4]

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.title.set_text('Solution')

    ax.plot(time_points, x_solution, label=solver.name)

    u0 = np.array([x0, y0, z0], dtype='longdouble')
    du_dt0 = np.array([vx0, vy0, vz0], dtype='longdouble')
    solver1 = EverhartIIRadau7ODESolver(f2, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=True,
                                        tolerance=tolerance)
    solution1, d_solution1, time_points1 = solver1.solve(print_benchmark=True, benchmark_name=solver1.name)
    x_solution1 = solution1[:, 0]
    y_solution1 = solution1[:, 1]
    z_solution1 = solution1[:, 2]

    ax.plot(time_points1, x_solution1, label=solver1.name)

    points_number = int((tmax - t0) / dt0)
    time_exact = np.linspace(t0, t0 + dt0 * points_number, (points_number + 1))
    x_exact, y_exact = exact_f(time_exact)
    plt.plot(time_exact, x_exact, label='Exact analytical solution')

    ax.legend(loc='upper left')

    error = abs(x_exact - x_solution)
    error1 = abs(x_exact - x_solution1)

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.title.set_text('Error')
    ax1.plot(error, label=solver.name)
    ax1.plot(error1, label=solver1.name)
    ax1.legend(loc='upper left')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()
```
____
