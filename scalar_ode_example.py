import math

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from cleanode.ode_solvers import *
import scipy.constants as const

# Example of the scalar ODE solving: capital growth
if __name__ == '__main__':

    # noinspection PyUnusedLocal
    def f(u: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Calculating the right side of the 1st order ODE
        :param u: variable value
        :type u: np.ndarray
        :param t: time
        :type t: np.ndarray
        :return: calculated value of the right part
        :rtype: np.ndarray
        """

        # Mathematically, the ODE system looks like this:
        # d(dx)/dt = t
        # d(dy)/dt = -g

        g = const.g

        vx = u[0]
        vy = u[1]

        right_sides = np.array([
            0,
            -g
        ], dtype='longdouble')

        return right_sides

    # noinspection PyUnusedLocal
    def f2(u: np.longdouble, du_dt: np.longdouble, t: np.ndarray) -> np.ndarray:
        """
        Calculating the right side of the 2nd order ODE
        :param u: variable
        :type u: np.longdouble
         :param du_dt: time derivative of variable
        :type du_dt: np.longdouble
        :param t: time
        :type t: np.ndarray
        :return: calculated value of the right part
        :rtype: np.ndarray
        """

        # Mathematically, the ODE looks like this:
        # d(dx)/dt = -g

        g = const.g

        right_side = - g

        return right_side


    # noinspection PyUnusedLocal
    def f2_1(u: np.longdouble, du_dt: np.longdouble, t: np.ndarray) -> np.ndarray:
        """
        Calculating the right side of the 2nd order ODE
        :param u: variable
        :type u: np.longdouble
         :param du_dt: time derivative of variable
        :type du_dt: np.longdouble
        :param t: time
        :type t: np.ndarray
        :return: calculated value of the right part
        :rtype: np.ndarray
        """

        # Mathematically, the ODE looks like this:
        # d(dx)/dt = x

        x = u[0]

        right_side = x

        return right_side

    # noinspection PyUnusedLocal
    def f_1(u: List[float], t: Union[np.ndarray, np.float64]) -> List:
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
        # dx/dt = V
        # dV/dt = t

        x = u[0]
        v = u[1]

        right_sides = [
            v,
            x,
        ]

        return right_sides


    # calculation parameters:
    t0 = np.longdouble(0)
    tmax = np.longdouble(3)
    dt0 = np.longdouble(0.3)

    points_number = int((tmax - t0) / dt0)
    time_exact = np.linspace(t0, tmax, points_number * 10)

    # u0 = np.array([1.0], dtype='longdouble')

    # solver = RungeKutta4ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u1, t1 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t1, u1, label=solver.name)
    # x_exact = np.exp(time_exact)

    # solver = EverhartIRadau7ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u2, t2 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t2, u2, label=solver.name)
    # x_exact = np.exp(time_exact)

    # x0 = 0
    # v0 = 0
    # u0 = np.array([x0], dtype='longdouble')  # начальное положение
    # du_dt0 = np.array([v0], dtype='longdouble')  # начальная скорость
    # solver = EverhartIIRadau7ODESolver(f2, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=False)
    # # solver = EverhartIILobatto21ODESolver(f2, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=False)
    # u3, t3 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t3, u3, label=solver.name)
    # x_exact = x0 + v0 * time_exact - const.g * time_exact ** 2 / 2

    # plt.plot(time_exact, x_exact, label='Exact analytical solution')

    x0 = 1
    v0 = 1

    u0 = np.array([x0], dtype='longdouble')  # начальное положение
    du_dt0 = np.array([v0], dtype='longdouble')  # начальная скорость
    solver = EverhartIIRadau7ODESolver(f2_1, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=False)
    u3, t3 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    plt.plot(t3, u3, label=solver.name)

    u0 = np.array([x0, v0], dtype='longdouble')
    solver = RungeKutta4ODESolver(f_1, u0, t0, tmax, dt0, is_adaptive_step=False)
    u1, t1 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    solution_x = u1[:, 0]
    plt.plot(t1, solution_x, label=solver.name)

    c1 = x0 - (x0 - v0) / 2
    c2 = (x0 - v0) / 2
    x_exact = c1 * np.exp(time_exact) + c2 / np.exp(time_exact)
    plt.plot(time_exact, x_exact, label='Exact analytical solution')


    plt.legend()

    plt.show()
