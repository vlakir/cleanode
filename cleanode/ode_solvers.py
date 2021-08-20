import numpy as np
from typing import Callable, Tuple
from typing import Union, List
from funnydeco import benchmark


class GenericExplicitRKODESolver:
    """
    Корневой класс, реализующий решение ДУ явными методами Рунге-Кутты
    """
    def __init__(self, f: Callable,
                 u0: Union[List, float],
                 t0: float,
                 tmax: float,
                 dt0: float,
                 butcher_tableau=None,
                 name='имя метода не определено',
                 is_adaptive_step=False):
        """
        :param f: функция вычисления правой части ДУ или правых частей системы уравнений
        :type f: Callable
        :param u0: начальное условие
        :type u0:  Union[List, float]
        :param t0: нижний предел интегрирования
        :type t0: float
        :param tmax: верхний предел интегрирования
        :type tmax: float
        :param dt0: начальный шаг интегрирования
        :type dt0: float
        :param butcher_tableau: таблица Бутчера
        :type butcher_tableau: np.array
        :param name: название метода
        :type name: str
        :param is_adaptive_step: использовать адаптивный шаг по времени
        :type is_adaptive_step: bool
        """

        self.f = f
        self.name = name
        self.butcher_tableau = butcher_tableau

        if self.butcher_tableau is None:
            raise ValueError('Не задана таблица Бутчера')

        if len(self.butcher_tableau[0]) == len(self.butcher_tableau):
            self.c = butcher_tableau[:-1, 0]
            self.b = butcher_tableau[-1, 1:]
            self.b1 = None
            self.a = butcher_tableau[:-1, 1:]
        elif len(self.butcher_tableau[0]) == len(self.butcher_tableau) - 1:  # есть дополнительная строка b1 для
            # проверки точности решения на шаге
            self.c = butcher_tableau[:-2, 0]
            self.b = butcher_tableau[-2, 1:]
            self.b1 = butcher_tableau[-1, 1:]
            self.a = butcher_tableau[:-2, 1:]
        else:
            raise ValueError('Некорректный формат таблицы Бутчера')

        if (np.count_nonzero(np.triu(self.a))) > 0:
            raise ValueError('В верхнем треугольнике матрицы a в таблице Бутчера есть ненулевые элементы. '
                             'Это недопустимо для явного метода Рунге-Кутты.')

        self.is_adaptive_step = is_adaptive_step

        self.u = None
        self.n = None
        self.dt = dt0
        self.tmax = tmax
        self.t0 = tmax
        self.dt0 = t0

        self.t = np.array([t0])

        if isinstance(u0, (float, int)):  # одиночное ДУ
            u0 = np.float(u0)
            self.ode_system_size = 1
        else:  # система ДУ
            u0 = np.asarray(u0)
            self.ode_system_size = u0.size  # количество ДУ в системе

        self.u0 = u0

    @benchmark
    def solve(self, print_benchmark=False, benchmark_name='') -> Tuple[np.ndarray, np.ndarray]:
        """
        Решение ДУ
        :param print_benchmark: выводить в консоль время выполнения
        :type print_benchmark: bool
        :param benchmark_name: имя бенчмарка для вывода в консоль
        :type benchmark_name: str
        :return: решение, время
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # патч, чтобы PyCharm не ругался:
        __ = print_benchmark
        __ = benchmark_name

        if self.ode_system_size == 1:  # scalar ODEs
            self.u = np.array([self.u0])
        else:  # systems of ODEs
            self.u = np.zeros((1, self.ode_system_size))
            self.u[0] = self.u0

        i = 0
        while self.t[i] <= self.tmax:
            self.n = i
            u_next = self._step()
            self.u = np.vstack([self.u, u_next])
            i += 1

        if self.is_adaptive_step:
            # 2do: для адаптивного шага реализовать финальную интерполяцию на равномерную сетку с шагом dt0
            pass

        return self.u, self.t

    def _step(self) -> np.ndarray:
        """
        Решение для одного шага интегрирования
        :return: решение
        :rtype: float
        """
        u, f, n, t, dt, a, b, c = self.u, self.f, self.n, self.t, self.dt, self.a, self.b, self.c

        k = np.zeros((len(b), self.ode_system_size))

        k[0] = np.array(f(u[n], t))
        for i in range(1, len(k)):
            summ = np.float(0)
            for j in range(i + 1):
                summ += a[i, j] * k[j]
            k[i] = f(u[n] + dt * summ, t[n] + c[i] * dt)

        unew = np.sum((b * k.T).T, axis=0)
        unew *= dt
        unew += u[n]

        self.t = np.append(self.t, self.t[-1] + self.dt)

        self._change_dt()

        return unew

    def _change_dt(self) -> None:
        """
        Адаптивный алгоритм изменения шага по дополнительной строке self.b1 таблицы Бутчера
        """
        if self.is_adaptive_step:
            if self.b1 is None:
                raise ValueError('Значение is_adaptive_step==True не поддерживается для нерасширенной таблицы Бутчера')

            # 2do: реализовать
            raise ValueError('Значение is_adaptive_step==True пока не поддерживается')
            # self.dt = ...


class EulerODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Эйлера
    """
    butcher_tableau = np.array([

        [0,         0],

        [None,      1]

        ], dtype='float')

    name = 'Метод Эйлера'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class MidpointODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод деления отрезка пополам
    """
    butcher_tableau = np.array([

        [0,     0,      0],
        [1/2,   1/2,    0],

        [None,  0,      1]

        ], dtype='float')

    name = 'Метод деления отрезка пополам'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class RungeKutta4ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Рунге-Кутты 4-го порядка
    """
    butcher_tableau = np.array([

        [0,         0,      0,      0,      0],
        [1/2,       1/2,    0,      0,      0],
        [1/2,       0,      1/2,    0,      0],
        [1,         0,      0,      1,      0],

        [None,      1/6,    1/3,    1/3,    1/6]

        ], dtype='float')

    name = 'Метод Рунге-Кутты 4-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Fehlberg45Solver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Рунге-Кутты-Феленберга
    """
    butcher_tableau = np.array([

        [0,             0,              0,              0,              0,              0,              0],
        [1/4,           1/4,            0,              0,              0,              0,              0],
        [3/8,           3/32,           9/32,           0,              0,              0,              0],
        [12/13,         1932/2197,      -7200/2197,     7296/2197,      0,              0,              0],
        [1,             439/216,        -8,             3680/513,       -845/4104,      0,              0],
        [1/2,           -8/27,          2,              -3544/2565,     1859/4104,      -11/40,         0],

        [None,          16/135,         0,              6656/12825,     28561/56430,    -9/50,          2/55],
        [None,          25/216,         0,              1408/2565,      2197/4104,      -1/5,           0]

        ], dtype='float')

    name = 'Метод Рунге-Кутты-Феленберга'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Heun2ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Хойна 2 порядка
    """
    butcher_tableau = np.array([

        [0,       0,      0],
        [1,       1,      0],

        [None,    1/2,    1/2]

        ], dtype='float')

    name = 'Метод Хойна 2 порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Ralston2ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Рэлстона 2 порядка
    """
    butcher_tableau = np.array([

        [0,         0,      0],
        [2/3,       2/3,    0],

        [None,      1/4,    3/4]

        ], dtype='float')

    name = 'Метод Рэлстона 2 порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class RungeKutta3ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Рунге-Кутты 3-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1/2,     1/2,    0,      0],
        [1,       -1,     2,      0],

        [None,    1/6,    2/3,    1/6]

    ], dtype='float')

    name = 'Метод Рунге-Кутты 3-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Heun3ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Хойна 3-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1/3,     1/3,    0,      0],
        [2/3,     0,      2/3,    0],

        [None,    1/4,    0,      3/4]

        ], dtype='float')

    name = 'Метод Хойна 3-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Ralston3ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Рэлстона 3-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1/2,     1/2,    0,      0],
        [3/4,     0,      3/4,    0],

        [None,    2/9,    1/3,    4/9]

        ], dtype='float')

    name = 'Метод Рэлстона 3-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class SSPRK3ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий высокостабильный медот Рунге-Кутты 3-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1,       1,      0,      0],
        [1/2,     1/4,    1/4,    0],

        [None,    1/6,    1/6,    2/3]

        ], dtype='float')

    name = 'Высокостабильный медот Рунге-Кутты 3-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Ralston4ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Рэлстона 4-го порядка
    """
    butcher_tableau = np.array([

        [0,             0,             0,              0,             0],
        [0.4,           0.4,           0,              0,             0],
        [0.45573725,    0.29697761,    0.15875964,     0,             0],
        [1,             0.21810040,    -3.05096516,    3.83286476,    0],

        [None,          0.17476028,    -0.55148066,    1.20553560,    0.17118476]

        ], dtype='float')

    name = 'Метод Рэлстона 4-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Rule384ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод правила 3/8 4-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,       0,      0,      0],
        [1/3,     1/3,     0,      0,      0],
        [2/3,     -1/3,    1,    0,      0],
        [1,       1,       -1,     1,      0],

        [None,    1/8,     3/8,    3/8,    1/8]

        ], dtype='float')

    name = 'Метод правила 3/8 4-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class HeunEuler21ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Хойна-Эйлера 2-1-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,      0],
        [1,       1,      0],

        [None,    1/2,    1/2],
        [None,    1,      0]

        ], dtype='float')

    name = 'Метод Хойна-Эйлера 2-1-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Fehlberg21ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Феленберга 2-1-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,        0,          0],
        [1/2,     1/2,      0,          0],
        [1,       1/256,    255/256,    0],

        [None,    1/512,    255/256,    1/512],
        [None,    1/256,    255/256,    0]

        ], dtype='float')

    name = 'Метод Феленберга 2-1-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class BogackiShampine32ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Богацкого–Шампина 3-2-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,       0,      0,      0],
        [1/2,     1/2,     0,      0,      0],
        [3/4,     0,       3/4,    0,      0],
        [1,       2/9,     1/3,    4/9,    0],

        [None,    2/9,     1/3,    4/9,    0],
        [None,    7/24,    1/4,    1/3,    1/8]

        ], dtype='float')

    name = 'Метод Богацкого–Шампина 3-2-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class CashKarp54ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Кэша-Карпа 5-4-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,             0,          0,              0,               0,            0],
        [1/5,     1/5,           0,          0,              0,               0,            0],
        [3/10,    3/40,          9/40,       0,              0,               0,            0],
        [3/5,     3/10,          -9/10,      6/5,            0,               0,            0],
        [1,       -11/54,        5/2,        -70/27,         35/27,           0,            0],
        [7/8,     1631/55296,    175/512,    575/13824,      44275/110592,    253/4096,     0],

        [None,    37/378,        0,          250/621, 	     125/594,         0,            512/1771],
        [None,    2825/27648,    0,          18575/48384,    13525/55296,     277/14336,    1/4]

        ], dtype='float')

    name = 'Метод Кэша-Карпа 5-4-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class DormandPrince54ODESolver(GenericExplicitRKODESolver):
    """
    Класс, реализующий метод Дорманда-Принса 5-4-го порядка
    """
    butcher_tableau = np.array([

        [0,       0,             0,              0,             0,           0,                0,           0],
        [1/5,     1/5,           0,              0,             0,           0,                0,           0],
        [3/10,    3/40,          9/40,           0,             0,           0,                0,           0],
        [4/5,     44/45,         -56/15,         32/9,          0,           0,                0,           0],
        [8/9,     19372/6561,    -25360/2187,    64448/6561,    -212/729,    0,                0,           0],
        [1,       9017/3168,     -355/33,        46732/5247,    49/176,      -5103/18656,      0,           0],
        [1,       35/384,        0,              500/1113,      125/192,     -2187/6784,       11/84,       0],

        [None,    35/384,        0,              500/1113,      125/192,     -2187/6784,       11/84,       0],
        [None,    5179/57600,    0,              7571/16695,    393/640,     -92097/339200,    187/2100,    1/40]

        ], dtype='float')

    name = 'Метод Дорманда-Принса 5-4-го порядка'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)
