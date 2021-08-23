import numpy as np
from typing import Callable, Tuple
from typing import Union, List
from funnydeco import benchmark


class GenericExplicitRKODESolver:
    """
    Core class implements explicit Runge-Kutta methods
    """
    def __init__(self, f: Callable,
                 u0: Union[List, float],
                 t0: float,
                 tmax: float,
                 dt0: float,
                 butcher_tableau=None,
                 name='method name is not defined',
                 is_adaptive_step=False):
        """
        :param f: function for calculating right parts
        :type f: Callable
        :param u0: initial conditions
        :type u0: Union[List, float]
        :param t0: lower limit of integration
        :type t0: float
        :param tmax: upper limit of integration
        :type tmax: float
        :param dt0: initial step of integration
        :type dt0: float
        :param butcher_tableau: Butcher tableau
        :type butcher_tableau: np.array
        :param name: method name
        :type name: string
        :param is_adaptive_step: use adaptive time step
        :type is_adaptive_step: bool
        """

        self.f = f
        self.name = name
        self.butcher_tableau = butcher_tableau

        if self.butcher_tableau is None:
            raise ValueError('Butcher tableau is not defined')

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
            raise ValueError('Butcher tableau has invalid form')

        if (np.count_nonzero(np.triu(self.a))) > 0:
            raise ValueError('There are non-zero elements in the upper triangle of the matrix a in the Butcher tableau.'
                             ' It is not allowed for an explicit Runge-Kutta method.')

        self.is_adaptive_step = is_adaptive_step

        self.u = None
        self.n = None
        self.dt = dt0
        self.tmax = tmax
        self.t0 = tmax
        self.dt0 = t0

        self.t = np.array([t0])

        if isinstance(u0, (float, int)):  # scalar ODE
            u0 = np.float(u0)
            self.ode_system_size = 1
        else:  # ODE system
            u0 = np.asarray(u0)
            self.ode_system_size = u0.size

        self.u0 = u0

    # noinspection PyUnusedLocal
    @benchmark
    def solve(self, print_benchmark=False, benchmark_name='') -> Tuple[np.ndarray, np.ndarray]:
        """
        ODE solution
        :param print_benchmark: output the execution time to the console
        :type print_benchmark: bool
        :param benchmark_name: name of the benchmark to output
        :type benchmark_name: string
        :return: solution, time
        :rtype: Type[np.ndarray, np.ndarray]
        """

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
            # 2do: for the adaptive step: to implement the final interpolation on a uniform grid with dt0 step
            pass

        return self.u, self.t

    def _step(self) -> np.ndarray:
        """
        One-step integration solution
        :return: solution
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
        Adaptive algorithm for changing the step by an additional row self.b1 of the Butcher tableau
        """
        if self.is_adaptive_step:
            if self.b1 is None:
                raise ValueError('is_adaptive_step==True is not supported for a non-extended Butcher tableau')

            # 2do: to implement
            raise ValueError('is_adaptive_step==True is not supported yet')
            # self.dt = ...


class EulerODESolver(GenericExplicitRKODESolver):
    """
    Implements the Euler method
    """
    butcher_tableau = np.array([

        [0,         0],

        [None,      1]

        ], dtype='float')

    name = 'Euler method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class MidpointODESolver(GenericExplicitRKODESolver):
    """
    Implements the Explicit midpoint method
    """
    butcher_tableau = np.array([

        [0,     0,      0],
        [1/2,   1/2,    0],

        [None,  0,      1]

        ], dtype='float')

    name = 'Explicit midpoint method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class RungeKutta4ODESolver(GenericExplicitRKODESolver):
    """
    Implements the fourth-order Runge–Kutta method
    """
    butcher_tableau = np.array([

        [0,         0,      0,      0,      0],
        [1/2,       1/2,    0,      0,      0],
        [1/2,       0,      1/2,    0,      0],
        [1,         0,      0,      1,      0],

        [None,      1/6,    1/3,    1/3,    1/6]

        ], dtype='float')

    name = 'Fourth-order Runge–Kutta method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Fehlberg45Solver(GenericExplicitRKODESolver):
    """
    Implements the Fehlberg method
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

    name = 'Fehlberg method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Ralston2ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Ralston method
    """
    butcher_tableau = np.array([

        [0,         0,      0],
        [2/3,       2/3,    0],

        [None,      1/4,    3/4]

        ], dtype='float')

    name = 'Ralston method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class RungeKutta3ODESolver(GenericExplicitRKODESolver):
    """
    Implements the third-order Runge–Kutta method
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1/2,     1/2,    0,      0],
        [1,       -1,     2,      0],

        [None,    1/6,    2/3,    1/6]

    ], dtype='float')

    name = 'Third-order Runge–Kutta method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Heun3ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Heun third-order method
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1/3,     1/3,    0,      0],
        [2/3,     0,      2/3,    0],

        [None,    1/4,    0,      3/4]

        ], dtype='float')

    name = 'Heun third-order method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Ralston3ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Ralston third-order method
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1/2,     1/2,    0,      0],
        [3/4,     0,      3/4,    0],

        [None,    2/9,    1/3,    4/9]

        ], dtype='float')

    name = 'Ralston third-order method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class SSPRK3ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Third-order Strong Stability Preserving Runge-Kutta method
    """
    butcher_tableau = np.array([

        [0,       0,      0,      0],
        [1,       1,      0,      0],
        [1/2,     1/4,    1/4,    0],

        [None,    1/6,    1/6,    2/3]

        ], dtype='float')

    name = 'Third-order Strong Stability Preserving Runge-Kutta method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Ralston4ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Ralston fourth-order method
    """
    butcher_tableau = np.array([

        [0,             0,             0,              0,             0],
        [0.4,           0.4,           0,              0,             0],
        [0.45573725,    0.29697761,    0.15875964,     0,             0],
        [1,             0.21810040,    -3.05096516,    3.83286476,    0],

        [None,          0.17476028,    -0.55148066,    1.20553560,    0.17118476]

        ], dtype='float')

    name = 'Ralston fourth-order method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Rule384ODESolver(GenericExplicitRKODESolver):
    """
    Implements the 3/8-rule fourth-order method
    """
    butcher_tableau = np.array([

        [0,       0,       0,      0,      0],
        [1/3,     1/3,     0,      0,      0],
        [2/3,     -1/3,    1,    0,      0],
        [1,       1,       -1,     1,      0],

        [None,    1/8,     3/8,    3/8,    1/8]

        ], dtype='float')

    name = '3/8-rule fourth-order method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class HeunEuler21ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Heun–Euler method
    """
    butcher_tableau = np.array([

        [0,       0,      0],
        [1,       1,      0],

        [None,    1/2,    1/2],
        [None,    1,      0]

        ], dtype='float')

    name = 'Heun–Euler method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class Fehlberg21ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Fehlberg RK1(2) method
    """
    butcher_tableau = np.array([

        [0,       0,        0,          0],
        [1/2,     1/2,      0,          0],
        [1,       1/256,    255/256,    0],

        [None,    1/512,    255/256,    1/512],
        [None,    1/256,    255/256,    0]

        ], dtype='float')

    name = 'Fehlberg RK1(2) method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class BogackiShampine32ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Bogacki–Shampine method
    """
    butcher_tableau = np.array([

        [0,       0,       0,      0,      0],
        [1/2,     1/2,     0,      0,      0],
        [3/4,     0,       3/4,    0,      0],
        [1,       2/9,     1/3,    4/9,    0],

        [None,    2/9,     1/3,    4/9,    0],
        [None,    7/24,    1/4,    1/3,    1/8]

        ], dtype='float')

    name = 'Bogacki–Shampine method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class CashKarp54ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Cash-Karp method
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

    name = 'Cash-Karp method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)


class DormandPrince54ODESolver(GenericExplicitRKODESolver):
    """
    Implements the Dormand–Prince method
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

    name = 'Dormand–Prince method'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, butcher_tableau=self.butcher_tableau, name=self.name)
