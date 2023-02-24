import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify

# x = sp.Symbol('x')
# y = sp.Symbol('y')
# state = sp.Matrix([x, y])
# print(state)
# print(state.T)

# f_symbolic = state[0] + state[1]
# print(f_symbolic)

# # if we want to use numpy tools for variable, we need to add 'numpy' at the back
# f = lambdify(np.array(state.T), f_symbolic, 'numpy')
# print(f([1, 2]))

p, v, z = sp.symbols('p v z')
x  = sp.Matrix([p, v, z])

f_symbolic = sp.Matrix([x[1], x[2], x[0]])
g_symbolic = sp.Matrix([0, 0.5, 0])

cbf_symbolic = x[2] - 0.5 * x[1]
clf_symbolic = (x[1] - 0.5) **2

f = lambdify(np.array(x.T), f_symbolic, 'numpy')
g = lambdify(np.array(x.T), g_symbolic, 'numpy')
cbf = lambdify(np.array(x.T), cbf_symbolic, 'numpy')

dx_cbf_symbolic = sp.Matrix([cbf_symbolic]).jacobian(x)
lf_cbf_symbolic = dx_cbf_symbolic * f_symbolic
lg_cbf_symbolic = dx_cbf_symbolic * g_symbolic

lf_cbf = lambdify(np.array(x.T), lf_cbf_symbolic, 'numpy')
print(lf_cbf([1, 2, 3]))
print(lf_cbf([1, 2, 3])[0][0])
print(lf_cbf([1, 2, 3]).shape)
