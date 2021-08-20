import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import scipy.constants as const
from cleanode.ode_solvers import *


if __name__ == '__main__':
    """
    Пример решения системы ДУ (бросок мяча)
    """
    def f(u: List[float], t: Union[np.ndarray, np.float64]) -> List:
        """
        Вычисление правых частей системы уравнений
        :param u: значения переменных
        :type u: List[float]
        :param t: время
        :type t: Union[np.ndarray, np.float64]
        :return: вычисленные значения правых частей
        :rtype: np.ndarray
        """

        # Математически СДУ выглядит следующим образом:
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

        # патч, чтобы PyCharm не ругался:
        __ = t
        __ = x
        __ = y

        return right_sides

    # параметры расчета:
    t0 = 0
    tmax = 1
    dt0 = 0.01

    # начальные условия:
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

    # решение представляет собой массив формата [x, vx, y, vy], выводить будем функцию y(x)
    x_solution = solution[:, 0]
    y_solution = solution[:, 2]

    plt.plot(x_solution, y_solution)

    plt.show()


