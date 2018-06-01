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

# Defining equilibrium point and system parameters
if n == 1:
    param_values = config.parameter_values_simple_pendulum
    x0 = [0., 0., 0., 0.]

elif n == 2:
    param_values = config.parameter_values_double_pendulum
    x0 = [0., 0., 0., 0., 0., 0.]

elif n == 3:
    param_values = config.parameter_values_triple_pendulum
    x0 = [0., 0., 0., 0., 0., 0., 0., 0.]

param_symb = list(a + d + m + J + (g, f))
param_list = zip(param_symb, param_values)

u0 = [0.0]
equilibrium_point = x0 + u0

# linearization of model @ equilibrium point
A, B = linearize_state_equ(fx, gx, q, qdot, u, param_list, equilibrium_point)

# lqr control to find K (we need it as a boundry-value (Pe) for our tracking !)
Q = np.identity(4)
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

Pe = results[1].reshape(16)

# lqr tracking of the desired trajectory :
# determining linearized model at any equilibrium point
A_eq, B_eq = linearize_state_equ(fx, gx, q, qdot, u, param_list)

# solving riccati differential equations inverse in time :
dynamic_symbs = q + qdot + [u]

frames_per_sec = 15
final_time = 2
t = np.linspace(0.0, final_time, final_time * frames_per_sec)

# P_dot= riccati_diff_equ(Pe, final_time, A_eq, B_eq, Q, R, dynamic_symbs)
Pe0 = np.identity(4)
# converting Matrix to Vector to be able to use in odeint
#   we're converting it back again to a Matrix ! (it also need some changes
#   in riccati_diff_equ function !)
Pe0 = Pe0.reshape(16)
# logger.debug('Pe = y0 : %f', Pe0)
# print ('Pe = y0 :', Pe0)
'''
P = odeint(riccati_diff_equ, Pe, t[::-1], args=(A_eq, B_eq, Q, R, dynamic_symbs))

with open('P_matrix.pkl','wb' ) as file :
    dill.dump(P, file)

'''
with open('P_matrix.pkl', 'rb') as file:
    P = dill.load(file)
# finding gain k for Tracking :

Psim = P[::-1]
K_matrix = generate_gain_matrix(R, B_eq, Psim, t, dynamic_symbs)

# finding states of the system using calculated K_matrix and
# comparing the results with desired trajecory !
xdot_func = sympy_states_to_func(dynamic_symbs, param_list)
ipydex.IPS()


def ode_function(x, t, xdot_func, K_matrix, Vect, mode='Closed_loop'):
    '''
    it's the dx/dt=func(x, t, args)  to be used in odeint
    (the first two arguments are system state x and time t)

    there are two modes available:
     - Closed_loop is defualt and can be used for tracking
     - Open_loop could be activated by setting  mode='Open_loop'
    

    ATTENTION :
      - use sympy_states_to_func to produce xdot functions out of 
        sympy expresisons. 
        (you have to run sympy_state_to_func once and store the result
        so you could pass it as xdot_func )

    '''
    if t > Vect[-1]:
        t = Vect[-1]

    # logging.debug('x_new: %s \n \n', x)
    # logging.debug('Debugging Message from ode_function')
    # logging.debug(
    #     '----------------------------------------------------------------')
    # n=len(Vect)
    xs = config.cs_ret[0](t)
    us = config.cs_ret[1](t)
    sys_dim= len(xs)  
    if mode == 'Closed_loop':
        k_list= [np.interp(t, Vect, K_matrix[:, i]) for i in range(sys_dim) ]
        k = np.array(k_list)
        delta_x = x - xs
        delta_u = (-1) * k.T.dot(delta_x)
        inputs = us + delta_u
        # loggings :
        # logging.debug('k :%s', k)
        # logging.debug('delta_x: %s', delta_x)
        # logging.debug('delta_u: %s \n', delta_u)
    elif mode == 'Open_loop':
        inputs = us

    state = x
    # logging.debug('t: %s \n', t)

    # logging.debug('us: %s', us)
    # logging.debug('xs:%s \n', xs)
    # logging.debug('state: %s', state)
    # logging.debug('inputs: %s \n', inputs)

    xdot = xdot_func(state, inputs)
    # logging.debug('x_current: %s', x)
    # logging.debug('xdot : %s ', xdot)

    return xdot


x_closed_loop = odeint(ode_function, xa, t, args=(xdot_func, K_matrix, t))
# x_open_loop= odeint(ode_function, xa, t, args=(xdot_func, K_matrix, t, 'Open_loop') )
ipydex.IPS()

xs = np.array([config.cs_ret[0](time).tolist() for time in t])

import matplotlib as mpl
import matplotlib.pyplot as plt

fig, axes = plt.subplots()
axes.plot(t, x_closed_loop[:, 1] * 180 / np.pi, 'o')
axes.plot(t, xs[:, 1] * 180 / np.pi)
# axes.plot(vect, K)

plt.show()