import numpy as np
import casadi as ca 


class OptionClass:
    """ 
    To get the parameter of the qP
    [value, default_value, type of value]
    """
    def __init__(self) -> None:
        self.options = None
        self.solve_Name = 'None'

    def set_option(self, key, value):
        try:
            if type(value) is self.options[key][2]:
                self.options[key][0] = value
            else:
                print(f"The type of value for the keyword '{key}' should be '{self.options[key][2]}'.")
        except:
            raise ValueError('Incorrect option keyword or type: ' + key)

    def get_option(self, key):
        try:
            value = self.options[key][0]
            return value
        except:
            raise ValueError('Incorrect option keyword: ' + key)

    def reset_option(self, key):
        try:
            self.options[key][0] = self.options[key][1]
        except:
            raise ValueError('Incorrect option keyword: ' + key)


class CbfClfQpOptions(OptionClass):
    def __init__(self) -> None:
        super().__init__()
        self.solve_Name = 'CBF-CLF-QP'
        self.setup()

    def setup(self):
        self.options = {
            # [Current value, default value, type]
            'u_max': [None, None, np.ndarray],
            'u_min': [None, None, np.ndarray],
            'clf_lambda': [None, 5, float],
            'cbf_gamma': [None, 5, float],
            'weight_input': [None, None, np.ndarray],
            'weight_slack': [None, 2e-2, float],
        }


class CBF_CLF_Qp:
    """
    This is the implementation of the CBF-CLF-QP method. The optimization problem is:

            min (u-u_ref).T * H * (u-u_ref) + p * delta**2
            s.t. L_f V(x) + L_g V(x) * u + lambda * V(x) <= delta  ---> CLF constraint
                 L_f B(x) + L_g B(x) * u + gamma * B(x) >= 0  ---> CBF constraint

    Input:
    :param  system  :   The dynamics of the system, containing CBF, CLF, and their Lie derivatives
    :param  x       :   The current state x
    :param  u_ref   :   The reference control input
    :param  slack   :   The slack activated or not, 1 -> activate while 0 -> not activate
    """
    def __init__(self, system, option_class) -> None:

        # dimension
        self.x_dim = system.x_dim
        self.u_dim = system.u_dim

        # CBF
        self.cbf = system.cbf
        self.lf_cbf = system.lf_cbf
        self.lg_cbf = system.lg_cbf

        # CLF
        self.clf = system.clf
        self.lf_clf = system.lf_clf
        self.lg_clf = system.lg_clf

        # get the parameter from the options
        self.weight_input = np.atleast_2d(option_class.get_option('weight_input'))
        self.weight_slack = option_class.get_option('weight_slack')

        self.clf_lambda = option_class.get_option('clf_lambda')
        self.cbf_gamma = option_class.get_option('cbf_gamma')

        # u_min and u_max
        self.u_max = option_class.get_option('u_max')
        if self.u_max.shape != (self.u_dim,):
            raise ValueError('The size of u_max should be udim-by-, a one dimensional vector in python.')
        self.u_min = option_class.get_option('u_min')
        if self.u_min.shape != (self.u_dim,):
            raise ValueError('The size of u_min should be udim-by-, a one dimensional vector in python.')

        # optimization function
        self.opti = ca.Opti()
        self.obj = None
        self.H = None
        
        # optimize variable
        self.u = self.opti.variable(self.u_dim)
        self.slack = None
        self.feasible = None

        # solver
        opts_setting = {
                'ipopt.max_iter': 100,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
        self.opti.solver('ipopt', opts_setting)

    def cbf_clf_qp(self, x, u_ref=None, with_slack=True):
        """
        Input
        :param x         :   The current state
        :param u_ref     :   A real number of 1D vector with shape (udim,)
        :param with_slack:   Indicator if there is slack variable

        return:
        :param u         :   The actual control
        :param slack     :   The value of slack variable
        :param clf       :   The value of clf
        :param cbf       :   The value of cbf
        :param feasilble :   If this Qp is feasible
        """
        if u_ref is None:
            u_ref = np.zeros(self.u_dim)
        else:
            if u_ref.shape != (self.u_dim,):
                raise ValueError(f'u_ref should have the shape size (u_dim,), now it is {u_ref.shape}')

        # build the weight function H in the cost function
        if self.weight_input.shape == (1, 1):
            # Weight input is a scalar
            self.H = self.weight_input * np.eye(self.u_dim)

        elif self.weight_input.shape == (self.u_dim, 1):
            # Weight_input is a vector, use it to form the diagonal of the H matrix
            self.H = np.diag(self.weight_input)

        elif self.weight_input.shape == (self.u_dim, self.u_dim):
            # Weight_input is a udim * udim matrix
            self.H = np.copy(self.weight_input)
        else:
            # unit array
            self.H = np.eye(self.u_dim)

        clf = self.clf(x)
        # shape in (1, 1)
        lf_clf = self.lf_clf(x)
        # shape in (1, m)
        lg_clf = self.lg_clf(x)

        cbf = self.cbf(x)
        lf_cbf = self.lf_cbf(x)
        lg_cbf = self.lg_cbf(x)

        # empty the constraint set
        self.opti.subject_to()

        if with_slack:
            self.slack = self.opti.variable()

            # object function
            self.obj = .5 * (self.u - u_ref).T @ self.H @ (self.u - u_ref)
            self.obj = self.obj + self.weight_slack * self.slack ** 2
            self.opti.minimize(self.obj)

            # constraint
            self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))
            self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))

            # LfV + LgV * u + lambda * V <= slack
            # LfB + LgB * u + gamma * B  >= 0
            self.opti.subject_to(lf_clf[0][0] + (lg_clf @ self.u)[0][0] + self.clf_lambda * clf - self.slack <=0)
            self.opti.subject_to(-lf_cbf[0][0] - (lg_cbf @ self.u)[0][0] - self.cbf_gamma * cbf <= 0)

            # optimize the Qp problem
            try:
                sol = self.opti.solve()
                self.feasible = True
            except:
                print(self.opti.return_status())
                self.feasible = False

            u = sol.value(self.u)
            slack = sol.value(self.slack)
            return u, slack, cbf, clf, self.feasible
        else:
            # object function
            self.obj = .5 * (self.u - u_ref).T @ self.H @ (self.u - u_ref)
            self.opti.minimize(self.obj)

            # constraint
            self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))

            # LfV + LgV * u + lambda * V <= slack
            # LfB + LgB * u + gamma * B  >= 0
            self.opti.subject_to(lf_clf + lg_clf * self.u + self.clf_lambda * clf <=0)
            self.opti.subject_to(lf_cbf + lg_cbf * self.u + self.cbf_gamma * cbf >= 0)
            
            # optimize the Qp problem
            try:
                sol = self.opti.solve()
                self.feasible = True
            except:
                print(self.opti.return_status())
                self.feasible = False

            u = sol.value(self.u)
            slack = None

            return u, slack, cbf, clf, self.feasible