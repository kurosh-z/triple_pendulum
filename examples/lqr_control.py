#IMPORTS
from eq_of_motion import *

import numpy as np
import sympy as sm
from numpy import deg2rad, rad2deg
from scipy import linalg as la
from sympy.physics.mechanics import  msubs
from scipy.integrate import odeint
from pydy.codegen.ode_function_generators import generate_ode_function
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

from matplotlib import pyplot as plt

#LQR control :


def lqr(A, B, Q, R):
    """
    solve the continous time lqr controller:
    dx/dt = A x + B u
    cost : integral x.T*Q*x + u.T*R*u

    """
    #solving the algebric riccati equation
    P = np.array([la.solve_continuous_are(A, B, Q, R)])

    #computer LQR gain
    K = np.array(la.inv(R) * (B.T * P))
    eigVals, eigVec = la.eig(A - B * K)

    return K, P, eigVals


def dlqr(A, B, Q, R):
    """
    solving discrete time lqr controller
    x[k+1] = A x[k] + B u[k]

    """
    #solving the algebric riccati equation
    P = np.array([la.solve_discrete_are(A, B, Q, R)])

    #computer LQR gain
    K = np.array([la.inv(R) * (B.T * P)])
    eigVals, eigVec = la.eig(A - B * K)

    return K, P, eigVals


#Constants
constants = [l, a[0], m[0], m[1], J[0], J[1], d[0], g]
specified=[f]

#generating ode functions from symbolic eq. :
right_hand_side = generate_ode_function(
    forcing_vector,
    q,
    qdot,
    constants,
    mass_matrix=mass_matrix,
    specifieds=specified)

#defining numerical constans :
numerical_constants = np.array([
    0.32,  # l0
    0.2,  #a0
    3.34,  #m0
    0.8512,  #m1
    0,  #J0
    0.01980,  #J1
    0.00715,  #d
    9.81  #g
])

#
numerical_specified = np.array([0])

#integrating the Eq. of Motion
x0 = np.array([0, deg2rad(45), 0, 0])
right_hand_side(x0, 0.0, numerical_specified, numerical_constants)

#integrating the Eq. of Motion
frames_per_sec = 60
final_time = 5.0
t = np.linspace(0.0, final_time, final_time * frames_per_sec)

parameter_dict = dict(zip(constants, numerical_constants))
#linearizing the equation at the point where pendulum on top of the cart is.q[1]=pi/2
linearizer = Kane.to_linearizer()


linearizer.r== sm.Matrix(specified) #definig input r as our specified
linearizer.r= sm.Matrix(specified)

#A, B = linearizer.linearize(A_and_B=True)
equilibrium_dict = {q[0]: 0, q[1]: np.pi / 2, qdot[0]:0, qdot[1]:0}

op_point=[equilibrium_dict, parameter_dict]
#A_op, B_op = linearizer.linearize(
#    A_and_B=True, op_point=op_point, simplify=True)

M, A, B= linearizer.linearize()
M_op=msubs(M, op_point)
A_op=msubs(A, op_point)
perm_mat=linearizer.perm_mat
A_lin=perm_mat.T*M_op.LUsolve(A_op)

