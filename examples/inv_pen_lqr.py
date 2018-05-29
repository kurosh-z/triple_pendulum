#IMPORTS
from inv_pen_trajectory import *

import numpy as np
import sympy as sm
from numpy import deg2rad, rad2deg
from scipy import linalg as linalg
from sympy.physics.mechanics import  msubs
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

#LQR control :


def lqr(A, B, Q, R):
    """
    solve the continous time lqr controller:
    dx/dt = A x + B u
    cost : integral x.T*Q*x + u.T*R*u

    """
    #solving the algebric riccati equation
    P = np.array([linalg.solve_continuous_are(A, B, Q, R)])

    #computer LQR gain
    K = np.array(linalg.inv(R) * (B.T * P))
    eigVals, eigVec = linalg.eig(A - B * K)

    return K, P, eigVals


def dlqr(A, B, Q, R):
    """
    solving discrete time lqr controller
    x[k+1] = A x[k] + B u[k]

    """
    #solving the algebric riccati equation
    P = np.array([linalg.solve_discrete_are(A, B, Q, R)])

    #computer LQR gain
    K = np.array([linalg.inv(R) * (B.T * P)])
    eigVals, eigVec = linalg.eig(A - B * K)

    return K, P, eigVals
