import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from cleanode.ode_solvers import *

# Example of the scalar ODE solving: capital growth
if __name__ == '__main__':

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
        # d(dx)/dt = x

        x = u[0]

        right_side = x

        return right_side

    # noinspection PyUnusedLocal
    def f1(u: List[float], t: Union[np.ndarray, np.float64]) -> List:
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
        # dV/dt = x

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
    dt0 = np.longdouble(0.0479)

    points_number = int((tmax - t0) / dt0)
    time_exact = np.linspace(t0, t0 + dt0 * points_number, (points_number + 1) * 10)

    x0 = 1
    v0 = 1

    u0 = np.array([x0], dtype='longdouble')  # начальное положение
    du_dt0 = np.array([v0], dtype='longdouble')  # начальная скорость
    solver = EverhartIIRadau7ODESolver(f2, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=True, tolerance=1e-8)
    u3, t3 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    plt.plot(t3, u3, label=solver.name)

    u0 = np.array([x0, v0], dtype='longdouble')
    solver = RungeKutta4ODESolver(f1, u0, t0, tmax, dt0, is_adaptive_step=True, tolerance=1e-8)
    u1, t1 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    solution_x = u1[:, 0]
    plt.plot(t1, solution_x, label=solver.name)

    c1 = x0 - (x0 - v0) / 2
    c2 = (x0 - v0) / 2
    x_exact = c1 * np.exp(time_exact) + c2 / np.exp(time_exact)
    plt.plot(time_exact, x_exact, label='Exact analytical solution')

    plt.legend()

    plt.show()
