# coding: utf-8
'''
-----------
To Do :
  - finding a general way to define Q and R according to number of pendulums
  - (should I have a dynamic way of changing  Q and R to find the best result ?! )
  - find the better ways compatible with ptyhon2 of calculating things generally 
    every n
  - logging should be changed (maybe to a file instead of consol!)
  -  visualization should be wiritten for genereal n pendulum 
'''
#=============================================================
# Standard Python modules
#=============================================================
import os, sys
import dill
# import logging

#=============================================================
# External Python modules
#=============================================================
#from __future__ import division, print_function
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from numpy.linalg import inv as np_inv
import scipy as sc
from scipy.integrate import odeint

from odeintw import odeintw

import ipydex
#=============================================================
# Standard Python modules
#=============================================================
from functions import *
from sys_model import *
from traj_opt import *
#=============================================================
# Lqr control for top equilibrium point
#=============================================================

# logging.basicConfig(
#     filename='pen_odeint.log',
#     level=logging.CRITICAL,
#     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

# logger = logging.getLogger()
# handler = logging.StreamHandler()
# formatter = logging.Formatter(
#     '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)

ipydex.activate_ips_on_exception()
# defining equilibrium point for system linearization at top
x0=[ 0.0 for i in range(2*len(q))]
u0 = [0.0]
equilibrium_point = x0 + u0

# linearization of model @ equilibrium point
A, B = linearize_state_equ(fx, gx, q, qdot, u, param_list, equilibrium_point)
print('linearized model is ready')
# lqr control to find K (we need it as a boundry-value (Pe) for our tracking !)
Q = np.identity(2*len(q))
R = 0.01 * np.identity(1)
# k_top = lqr(A, B, Q, R)
results = lqr(A, B, Q, R, additional_outputs=True)
k_top = results[0]
'''
# k(t)= R^-1 * B.T * P(t)  ---> Pe=(B.T).inv() * R * k_top
# ATTENTION :
#             - B is a 4*1 matrix so we couldnt find Pe exactly using
#               equation above! We impose a second condition for Pe
#               to be a diagonal Matrix. --> Pe=diag(p1, p2, p3, p4)
#
#             - On the other hand because the first and second
#               component of the vector B are always 0 , finding p1, p2
#               provide no additional informations. so we could set
#               them to be 0 and just find p3 and p4 !

diag_Pe= [0.0 for i in range(len(q))]
for i in range(len(q)):
      indx=i+len(q)
      diag_Pe.append(k_top[0,indx]*R[0,0]/B[indx,0])

Pe=np.diag(diag_Pe)
'''

Pe = results[1].reshape((2*len(q))**2)

# lqr tracking of the desired trajectory :
# determining linearized model at any equilibrium point
A_eq, B_eq = linearize_state_equ(fx, gx, q, qdot, u, param_list)

# solving riccati differential equations inverse in time :
dynamic_symbs = q + qdot + [u]

frames_per_sec = 15
final_time = 2
t = np.linspace(0.0, final_time, final_time * frames_per_sec)



print('Integrating riccati differential equations to find P matrix :') 
P = odeint(
    riccati_diff_equ, Pe, t[::-1], args=(A_eq, B_eq, Q, R, dynamic_symbs))

if n == 1:
    with open('P_matrix.pkl', 'wb') as file:
        dill.dump(P, file)
elif n == 2:
    with open('P_matrix_double.pkl', 'wb') as file:
        dill.dump(P, file)
elif n == 3:
    with open('P_matrix_triple.pkl', 'wb') as file:
        dill.dump(P, file)
'''
if n == 1 :
    with open('P_matrix.pkl', 'rb') as file:
        P = dill.load(file)
if n== 2 :
    with open('P_matrix_double.pkl', 'rb') as file:
        P = dill.load(file)
if n== 3 :
    with open('P_matrix_triple.pkl', 'rb') as file:
        P = dill.load(file)   
'''


# finding gain k for Tracking :

print('generating gain_matrix using P')
Psim = P[::-1]
K_matrix = generate_gain_matrix(R, B_eq, Psim, t, dynamic_symbs)
print('gain matrix is ready!')
ipydex.IPS()
# finding states of the system using calculated K_matrix and
# comparing the results with desired trajecory !
xdot_func = sympy_states_to_func(dynamic_symbs, param_list)

print('integrating to find x_closed_loop')
x_closed_loop = odeint(ode_function, xa, t, args=(xdot_func, K_matrix, t))
# x_open_loop= odeint(ode_function, xa, t, args=(xdot_func, K_matrix, t, 'Open_loop') )

xs = np.array([config.cs_ret[0](time).tolist() for time in t])

import matplotlib as mpl
import matplotlib.pyplot as plt

fig, axes = plt.subplots()
axes.plot(t, x_closed_loop[:, 1] * 180 / np.pi, 'o')
axes.plot(t, xs[:, 1] * 180 / np.pi)
# axes.plot(vect, K)

plt.show()
ipydex.IPS()