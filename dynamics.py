import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

class  AdaptiveCruiseControl:
    """
    Define the symbolic dynamic:    dx = f(x) + g(x) * u
    x contains 3 states:  p -> position     v -> velocity    relative z -> distance
    u contain 1 control: wheel force
    """
    def __init__(self, params) -> None:
        """
        The input 'params' is a dictionary type -> argument which contains the following parameters:

        :param f0   :   To define the rolling resistance;
        :param f1   :   To define the rolling resistance;
        :param f2   :   To define the rolling resistance;
        :param m    :   The mass;
        :param v0   :   The speed of leading cruise;
        :param T    :   The time horizon for lookahead;
        :param cd   :   The deceleration parameter in constraints of control;
        :param vd   :   The desired velocity in clf;
        :param G    :   The acceleration of gravity
        :param udim :   The dimension of control profile u
        """

        self.f0 = params['f0']
        self.f1 = params['f1']
        self.f2 = params['f2']
        self.v0 = params['v0']
        self.m  = params['m']

        self.T  = params['T']
        self.cd = params['cd']
        self.G  = params['G']
        self.u_dim = None

        if 'udim' in params.keys():
            self.u_dim = params['udim']
        else:
            print(f'The dimension of input u is not given, set it to be default 1')
            self.u_dim = 1

        # preference velocity
        self.vd = params['vd']

        # system states
        p, v, z = sp.symbols('p v z')
        self.x  = sp.Matrix([p, v, z])
        self.x_dim = self.x.shape[0]

        # reference control
        self.Fr = None

        # system dynamics and clf, cbf
        self.f = None
        self.f_symbolic = None

        self.g = None
        self.g_symbolic = None

        self.cbf = None
        self.cbf_symbolic = None

        # Lie derivative of cbf w.r.t f / g as a function
        self.lf_cbf = None
        self.lf_cbf_symbolic = None

        self.lg_cbf = None
        self.lg_cbf_symbolic = None

        self.clf = None
        self.clf_symbolic = None

        # Lie derivative of clf w.r.t f / g as a function
        self.lf_clf = None
        self.lf_clf_symbolic = None

        self.lg_clf = None
        self.lg_clf_symbolic = None

        # Define the symbolic expression for system dynamics, CLF and CBF in a symbolic way
        self.f_symbolic, self.g_symbolic = self.simple_car_dynamics()
        self.f = lambdify(np.array(self.x.T), self.f_symbolic, 'numpy')
        if self.f(np.ones(self.x_dim)).shape != (self.x_dim, 1):
            raise ValueError(f'The output of f(x) should be (xdim, 1), now it is {self.f(np.ones(self.x_dim)).shape}')

        self.g = lambdify(np.array(self.x.T), self.g_symbolic, 'numpy')
        if self.g(np.ones(self.x_dim)).shape != (self.x_dim, 1):
            raise ValueError(f'The output of g(x) should be (xdim, 1), now it is {self.g(np.ones(self.x_dim)).shape}')

        self.cbf_symbolic = self.define_cbf()
        self.cbf = lambdify(np.array(self.x.T), self.cbf_symbolic, 'numpy')

        self.clf_symbolic = self.define_clf()
        self.clf = lambdify(np.array(self.x.T), self.clf_symbolic, 'numpy')

        # get the Lie derivatives calculator
        self.lie_derivatives_calculator()

    def simple_car_dynamics(self):
        self.Fr = self.f0 + self.f1 * self.x[1] + self.f2 * self.x[1] ** 2
        f = sp.Matrix([self.x[1], -self.Fr / self.m, self.v0 - self.x[1]])
        g = sp.Matrix([0, 1/self.m, 0])
        return f, g

    def define_cbf(self):
        cbf = self.x[2] - self.T * self.x[1] - .5 * (self.x[1] - self.v0) ** 2 / (self.cd * self.G)
        return cbf

    def  define_clf(self):
        clf = (self.x[1] - self.vd) ** 2
        return clf

    def get_reference_control(self, x):
        return np.array([self.f0 + self.f1 * x[1] + self.f2 * x[1] ** 2])

    def lie_derivatives_calculator(self):
        # CBF
        dx_cbf_symbolic = sp.Matrix([self.cbf_symbolic]).jacobian(self.x)
        self.lf_cbf_symbolic = dx_cbf_symbolic * self.f_symbolic
        self.lg_cbf_symbolic = dx_cbf_symbolic * self.g_symbolic

        # lf_cbf: output is a 1 by 1 array   
        # lg_cbf: output is a 1 by u_dim array
        self.lf_cbf = lambdify(np.array(self.x.T), self.lf_cbf_symbolic, 'numpy')     
        self.lg_cbf = lambdify(np.array(self.x.T), self.lg_cbf_symbolic, 'numpy')   

        # CLF
        dx_clf_symbolic = sp.Matrix([self.clf_symbolic]).jacobian(self.x)
        self.lf_clf_symbolic = dx_clf_symbolic * self.f_symbolic
        self.lg_clf_symbolic = dx_clf_symbolic * self.g_symbolic

        # lf_clf: output is a 1 by 1 array   
        # lg_clf: output is a 1 by u_dim array
        self.lf_clf = lambdify(np.array(self.x.T), self.lf_clf_symbolic, 'numpy')
        self.lg_clf = lambdify(np.array(self.x.T), self.lg_clf_symbolic, 'numpy')

    def __str__(self) -> str:
        return f'Class contains the states {self.x}, \n' + \
                f'system dynamic f {self.f} and g {self.g} \n' \
                f'CLF {self.clf}, \n' + \
                f'CBF {self.cbf}, \n'
    
    def next_state(self, current_state, u, dt):
        next_state = current_state + dt * (self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.u_dim, -1)).T[0]) 

        return next_state
