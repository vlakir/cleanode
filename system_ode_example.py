import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import scipy.constants as const
from cleanode.ode_solvers import *


# Example of the system ODE solving: cannon firing
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
        # dVx/dt = 0
        # dy/dt = Vy
        # dVy/dt = -g

        g = const.g

        x = u[0]
        vx = u[1]
        y = u[2]
        vy = u[3]

        right_sides = [
            vx,
            0,
            vy,
            -g
            ]

        return right_sides

    # calculation parameters:
    t0 = 0
    tmax = 1
    dt0 = 0.01

    # initial conditions:
    x0 = 0
    y0 = 0
    v0 = 5
    angle_degrees = 80

    angle_radians = angle_degrees * np.pi / 180
    vx0 = v0 * np.cos(angle_radians)
    vy0 = v0 * np.sin(angle_radians)

    u0 = [x0, vx0, y0, vy0]

    solver = RungeKutta4ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)

    solution, time_points = solver.solve(print_benchmark=True, benchmark_name=solver.name)

    # the solution is an array of the format [x, vx, y, vy], we will output the function y(x)
    x_solution = solution[:, 0]
    y_solution = solution[:, 2]

    plt.plot(x_solution, y_solution)

    plt.show()
