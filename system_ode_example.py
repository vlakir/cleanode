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

    def exact_f(t):
        x = x0 + vx0 * time_exact
        y = y0 + vy0 * time_exact - const.g * time_exact ** 2 / 2
        return x, y

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
        # d(dx)/dt = 0
        # d(dy)/dt = -g

        g = const.g

        vx = u[0]
        vy = u[1]

        right_sides = np.array([
            0,
            -g
        ], dtype='longdouble')

        return right_sides


    # calculation parameters:
    t0 = np.longdouble(0)
    tmax = np.longdouble(3)
    dt0 = np.longdouble(0.3)

    # initial conditions:
    x0 = np.longdouble(0)
    y0 = np.longdouble(0)
    v0 = np.longdouble(5)
    angle_degrees = 80

    angle_radians = angle_degrees * np.pi / 180
    vx0 = v0 * np.cos(angle_radians)
    vy0 = v0 * np.sin(angle_radians)

    u0 = np.array([x0, vx0, y0, vy0], dtype='longdouble')
    solver = RungeKutta4ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    solution, time_points = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    x_solution = solution[:, 0]
    y_solution = solution[:, 2]
    plt.plot(x_solution, time_points, label=solver.name)

    # u0 = np.array([x0, vx0, y0, vy0], dtype='longdouble')
    # solver = EverhartIRadau7ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # solution, time_points = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # x_solution1 = solution[:, 0]
    # y_solution1 = solution[:, 2]
    # plt.plot(x_solution1, time_points, label=solver.name)
    #
    u0 = np.array([x0, y0], dtype='longdouble')
    du_dt0 = [vx0, vy0]
    solver = EverhartIIRadau7ODESolver(f2, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=False)
    solution, time_points = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    x_solution1 = solution[:, 0]
    y_solution1 = solution[:, 1]
    plt.plot(x_solution1, time_points, label=solver.name)

    points_number = int((tmax - t0) / dt0)
    time_exact = np.linspace(t0, tmax, points_number * 10)
    x_exact, y_exact = exact_f(time_exact)
    plt.plot(x_exact, time_exact, label='Exact analytical solution')

    plt.legend()
    plt.show()
