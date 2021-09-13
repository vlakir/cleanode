import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from cleanode.ode_solvers import *
import scipy.constants as const

# Example of the scalar ODE solving: capital growth
if __name__ == '__main__':

    # noinspection PyUnusedLocal
    def f(u: float, t: Union[np.ndarray, np.float64]) -> float:
        """
        Calculating the right side of the 1st order ODE
        :param u: variable value
        :type u: float
        :param t: time
        :type t: Union[np.ndarray, np.float64]
        :return: calculated value of the right part
        :rtype: float
        """

        # Mathematically, the ODE looks like this:
        # du/dt = u

        right_side = u

        return right_side

    # noinspection PyUnusedLocal
    def f2(u: np.longdouble, du_dt: np.longdouble, t: Union[np.ndarray, np.longdouble]) -> float:
        """
        Calculating the right side of the 2nd order ODE
        :param u: variable
        :type u: np.longdouble
         :param du_dt: time derivative of variable
        :type du_dt: np.longdouble
        :param t: time
        :type t: Union[np.ndarray, np.float64]
        :return: calculated value of the right part
        :rtype: float
        """

        # Mathematically, the ODE looks like this:
        # d(dx)/dt = -g

        g = const.g

        right_side = - g

        return right_side


    # calculation parameters:
    t0 = np.longdouble(0)
    tmax = np.longdouble(3)
    dt0 = np.longdouble(0.3)

    points_number = int((tmax - t0) / dt0)
    time_exact = np.linspace(t0, tmax, points_number * 10)

    # # initial condition:
    # u0 = 1
    # solver = RungeKutta4ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u_exact = np.exp(time_exact)

    # initial condition:
    u0 = np.longdouble(0.0)  # начальное положение
    du_dt0 = np.longdouble(16.2)  # начальная скорость
    solver = EverhartIIRadau21ODESolver(f2, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=False)
    # solver = EverhartIILobatto21ODESolver(f2, u0, du_dt0, t0, tmax, dt0, is_adaptive_step=False)

    u_exact = u0 + du_dt0 * time_exact - const.g * time_exact ** 2 / 2

    u3, t3 = solver.solve(print_benchmark=True, benchmark_name=solver.name)

    plt.plot(t3, u3, label=solver.name)

    plt.plot(time_exact, u_exact, label='Exact analytical solution')

    plt.legend()

    plt.show()
