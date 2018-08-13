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
import os
import sys
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

# from odeintw import odeintw

# import ipydex
#=============================================================
# Standard Python modules
#=============================================================
from myfuncs import *
import cfg

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

# ipydex.activate_ips_on_exception()


def tracking_control(ct):
    '''
    
    - linearize the system at top equilibrium point
    - finding feedback-control's gain at top equilibrium point
    - generate P_matrix 
    - generate gain for time varying control of the sysmtem
    - it needs the results form sys_model and traj_opt

    '''
    # fiding symbols we need
    dynamic_symbs = ct.model.dynamic_symbs
    q = ct.model.q
    u = ct.model.q
    fx = ct.model.fx
    gx = ct.model.gx
    max_time = ct.trajectory.max_time
    number_of_pendulums = ct.number_of_pendulums
    cs_ret = ct.trajectory.cs_ret
    xa = ct.trajectory.xa
    label = ct.label

    # defining equilibrium point for system linearization at top
    x0 = [0.0 for i in range(2 * len(q))]
    u0 = [0.0]
    equilibrium_point = x0 + u0

    # linearization of model @ equilibrium point
    A, B = linearize_state_equ(fx, gx, dynamic_symbs, equilibrium_point)
    print('linearized model is ready !')

    # lqr control to find K (we need it as a boundry-value (Pe) for our tracking !)
    Q = np.identity(2 * len(q))
    R = 0.01 * np.identity(1)

    # k_top = lqr(A, B, Q, R)
    results = lqr(A, B, Q, R, additional_outputs=True)
    k_top = results[0]

    # reshape P matrix and use it as initial guess for P
    Pe = results[1].reshape((2 * len(q))**2)

    # finding linearized model as a function of equilibrium point
    A_func, B_func = linearize_state_equ(
        fx, gx, dynamic_symbs, output_mode='numpy_func')

    # solving riccati differential equations inverse in time :
    frames_per_sec = 120
    final_time = max_time
    tvec = np.linspace(0.0, final_time, final_time * frames_per_sec)
    
    print('\n \n')
    print('========================Riccati differential equations========================')
    print('Integrating riccati differential equations to find P matrix :')
    print('==============================================================================')

    P = odeint(
        riccati_diff_equ,
        Pe,
        tvec[::-1],
        args=(A_func, B_func, Q, R, dynamic_symbs))

    if number_of_pendulums == 1:
        with open('P_matrix.pkl', 'wb') as file:
            dill.dump(P, file)

    elif number_of_pendulums == 2:
        with open('P_matrix_double.pkl', 'wb') as file:
            dill.dump(P, file)

    elif number_of_pendulums == 3:
        with open('P_matrix_triple.pkl', 'wb') as file:
            dill.dump(P, file)
    '''
    if number_of_pendulums == 1 :
        with open('P_matrix.pkl', 'rb') as file:
            P = dill.load(file)

    elif number_of_pendulums== 2 :
        with open('P_matrix_double.pkl', 'rb') as file:
            P = dill.load(file)

    elif number_of_pendulums== 3 :
        with open('P_matrix_triple.pkl', 'rb') as file:
            P = dill.load(file)   
    
    '''

    # finding gain k for Tracking :
    print('\n \n')
    print('======================== gain matrix ========================')
    print('generating gain_matrix using P')
    Psim = P[::-1]
    K_matrix = generate_gain_matrix(R, B_func, Psim, tvec, dynamic_symbs)
    print('gain matrix is ready!')

    # saving K_matrix in a numpy file
    np.save('K_matrix' + '_' + label + 'max_time_' + str(max_time) + '.npy',
            K_matrix)

    # ipydex.IPS()

    # finding states of the system using  K_matrix :
    
    xdot_func = sympy_states_to_func()
    print('\n \n')
    print('======================== x_closed_loop ========================')
    print('integrating to find x_closed_loop')

    # xa mit Abweichung !
    #xa = [0.0] + [np.pi/2
    #             for i in range(len(q) - 1)] + [0 for i in range(len(q))]

    x_closed_loop = odeint(
        ode_function, xa, tvec, args=(xdot_func, K_matrix, tvec))
    # x_open_loop= odeint(ode_function, xa, t, args=(xdot_func, K_matrix, t, 'Open_loop') )

    # saving x_closed_loop in a numpy file
    np.save('x_closed_loop' + '_' + label + '_'+ '_max_time_' + str(max_time) + '.npy',
            x_closed_loop)
    np.save('u_closed_loop' + '_' + label + '_'+ '_max_time_' + str(max_time) + '.npy',
            np.array(ct.tracking.ucl))        

    # returning the results :
    ct.tracking.x_closed_loop = x_closed_loop
    ct.tracking.tvec = tvec
    ct.tracking.P_matrix = P
    ct.tracking.gain_matrix = K_matrix
    ct.tracking.ucl=np.array(ct.tracking.ucl)

    
    
    # xs = np.array([cs_ret[0](time).tolist() for time in tvec])
    
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    
    # fig, axes = plt.subplots()
    # axes.plot(tvec, x_closed_loop[:, 1] * 180 / np.pi, 'o')
    # axes.plot(tvec, xs[:, 1] * 180 / np.pi)
    # # axes.plot(vect, K)
    
    # plt.show()


    # ipydex.IPS()
