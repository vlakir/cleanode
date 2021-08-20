import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import scipy.constants as const
from cleanode.ode_solvers import *


def example_one_ode_solving() -> None:
    """
    Пример решения ДУ разными методами (u' = u)
    """
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

    solver = EulerODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    u1, t1 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    plt.plot(t1, u1, label=solver.name)
    #
    # solver = MidpointODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u2, t2 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t2, u2, label=solver.name)
    #
    # solver = RungeKutta4ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u3, t3 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t3, u3, label=solver.name)
    #
    # solver = Fehlberg45Solver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u4, t4 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t4, u4, label=solver.name)
    #
    # solver = Heun2ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u5, t5 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t5, u5, label=solver.name)
    #
    # solver = Ralston2ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u6, t6 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t6, u6, label=solver.name)
    #
    # solver = RungeKutta3ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u7, t7 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t7, u7, label=solver.name)
    #
    # solver = Heun3ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u8, t8 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t8, u8, label=solver.name)
    #
    # solver = Ralston3ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u9, t9 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t9, u9, label=solver.name)
    #
    # solver = SSPRK3ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u10, t10 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t10, u10, label=solver.name)
    #
    # solver = Ralston4ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u11, t11 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t11, u11, label=solver.name)
    #
    # solver = Rule384ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u12, t12 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t12, u12, label=solver.name)
    #
    # solver = HeunEuler21ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u13, t13 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t13, u13, label=solver.name)
    #
    # solver = Fehlberg21ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u14, t14 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t14, u14, label=solver.name)
    #
    # solver = BogackiShampine32ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u15, t15 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t15, u15, label=solver.name)
    #
    # solver = CashKarp54ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # u16, t16 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    # plt.plot(t16, u16, label=solver.name)
    #
    solver = DormandPrince54ODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    u17, t17 = solver.solve(print_benchmark=True, benchmark_name=solver.name)
    plt.plot(t17, u17, label=solver.name)

    points_number = int((tmax - t0) / dt0)
    time_exact = np.linspace(t0, tmax, points_number * 10)
    plt.plot(time_exact, np.exp(time_exact), label='Точное решение')

    plt.legend()

    plt.show()


def example_system_ode_solving() -> None:
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
    # solver = EulerODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # solver = MidpointODESolver(f, u0, t0, tmax, dt0, is_adaptive_step=False)
    # solver = Fehlberg45Solver(f, u0, t0, tmax, dt0, is_adaptive_step=False)

    solution, time_points = solver.solve(print_benchmark=True, benchmark_name=solver.name)

    # решение представляет собой массив формата [x, vx, y, vy], выводить будем функцию y(x)
    x_solution = solution[:, 0]
    y_solution = solution[:, 2]

    plt.plot(x_solution, y_solution)

    plt.show()
