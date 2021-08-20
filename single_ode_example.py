import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import scipy.constants as const
from cleanode.ode_solvers import *

# Example of single ODE solving (capital growth)
if __name__ == '__main__':
    def f(u: float, t: Union[np.ndarray, np.float64]) -> float:
        """
        Вычисление правой части уравнения
        :param u: значение переменной
        :type u: float
        :param t: время
        :type t: Union[np.ndarray, np.float64]
        :return: вычисленное значение правой части
        :rtype: float
        """

        # Математически ДУ выглядит следующим образом:
        # du/dt = u

        right_side = u

        # патч, чтобы PyCharm не ругался:
        __ = t

        return right_side

    # параметры расчета:
    t0 = 0
    tmax = 3
    dt0 = 0.3

    # начальное условие:
    u0 = 1

    solver = RungeKutta4ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    u3, t3 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    plt.plot(t3, u3, label=solver.name)

    points_number = int((tmax - t0) / dt0)
    time_exact = np.linspace(t0, tmax, points_number * 10)
    plt.plot(time_exact, np.exp(time_exact), label='Точное решение')

    plt.legend()

    plt.show()

