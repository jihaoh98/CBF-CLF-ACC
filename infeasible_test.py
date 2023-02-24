import casadi as ca 
import numpy as np

opti = ca.Opti()
x = opti.variable()
y = opti.variable()

opti.subject_to(x + y == 1)
opti.subject_to(x + y == 2)
opti.minimize(x + y)

opts_setting = {
                'ipopt.max_iter': 100,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
opti.solver('ipopt', opts_setting)
try:
    sol = opti.solve()
    print(sol.value(x))
    print(sol.value(y))
except:
    print(opti.return_status())

if opti.return_status() == 'Infeasible_Problem_Detected':
    print('Infeasible!')
