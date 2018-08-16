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
from multiprocessing import Pool
import multiprocessing
# from pathos.multiprocessing import ProcessingPool as Pool
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
from myfuncs import convert_container_res_to_splines
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


def parallelized_tracking_control(ct, pool_size=3):
    '''

    - linearize the system at top equilibrium point
    - finding feedback-control's gain at top equilibrium point
    - generate P_matrix
    - generate gain for time varying control of the sysmtem
    - it needs the results form sys_model and traj_opt

    '''

    # fining key and values of pytrajecotory_res
    pytrajectory_res = ct.trajectory.pytrajectory_res
    traj_keys, traj_values = zip(*pytrajectory_res.items())
    # provide a dictionary for the results to be saved to !
    ct.tracking.tracking_res = dict([(traj_key, {}) for traj_key in traj_keys])
    # provide a dictionary for the xx and uu functions to be saved to !
    ct.trajectory.parallel_res = dict(
        [(traj_key, ()) for traj_key in traj_keys])

    # multi_args=[(ct, traj_key) for traj_key in traj_keys]
    processor_pool = Pool(pool_size)
    # ipydex.IPS()
    processor_pool.map(_tracking_control, [traj_keys[1]])

    # Process_jobs=[]
    # for key in traj_keys :
    #     p= multiprocessing.Process(target= _tracking_control, args=(key,))
    #     Process_jobs.append(p)
    #     p.start()
    #     p.join()
    """
    # finding key and values of trajectories
    trajectories = ct.trajectory.parallel_res
    traj_keys, traj_values = zip(*trajectories.items())


    # defining arg_keys and arg_values for argdict (to pass to _tracking_contorl)
    arg_keys = [('ct', 'traj_label') for i in range(len(trajectories))]
    arg_values = [(ct, traj_key) for traj_key in traj_keys]
    multi_args = [
        zip(arg_key, arg_value)
        for arg_key, arg_value in zip(arg_keys, arg_values)
    ]
    multi_arg_dicts = [dict(multi_arg) for multi_arg in multi_args]

    processor_pool = Pool(pool_size)
    # provoide a dictionary to save the results to
    ct.tracking.tracking_res = dict([(traj_key, {}) for traj_key in traj_keys])
    # parallel processing of _tracking_control
    processor_pool.map(_tracking_control, multi_arg_dicts)
    """

    return


def _tracking_control(arg_tupel):
    '''
    arg_tupel : tupel contains (ct, traj_label)
    '''

    # ct = arg_tupel[0]
    ct = cfg.pendata
    # print(dir(ct.tracking))
    # traj_label = arg_tupel[1]
    traj_label = arg_tupel

    final_time = float(traj_label.split("_")[1])

    # dynamic_symbs = ct.model.dynamic_symbs
    # A_func = ct.tracking.A_func
    # B_func = ct.tracking.B_func
    # Pe = ct.tracking.Pe
    # Q = ct.tracking.Q
    # R = ct.tracking.R
    # xa = ct.trajectory.xa
    label = ct.label

    # fiding symbols we need
    dynamic_symbs = ct.model.dynamic_symbs
    q = ct.model.q
    u = ct.model.q
    fx = ct.model.fx
    gx = ct.model.gx

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

    # ct.tracking.k_top = k_top
    # ct.tracking.Pe = Pe
    # ct.tracking.A_func = A_func
    # ct.tracking.B_func = B_func
    # ct.tracking.Q = Q
    # ct.tracking.R = R

    # provide a list for u_closed_loop to be saved to !
    # ipydex.IPS()
    # print('label',traj_label)
    ct.tracking.tracking_res[traj_label].update({'ucl': []})

    # convert containerized results of pytrajectory to functions!
    convert_container_res_to_splines(ct, traj_label)

    # solving riccati differential equations inverse in time :
    frames_per_sec = 240
    tvec = np.linspace(0.0, final_time, final_time * frames_per_sec)

    print('\n \n')
    print(
        '========================Riccati differential equations========================'
    )
    print('Integrating riccati differential equations to find P matrix :')
    print(
        '=============================================================================='
    )

    P = odeint(
        riccati_diff_equ,
        Pe,
        tvec[::-1],
        args=(A_func, B_func, Q, R, dynamic_symbs, traj_label))

    with open('P_matrix_' + label + traj_label + '.pkl', 'wb') as file:
        dill.dump(P, file)
    '''
    with open('P_matrix_'+ label+traj_label+'.pkl', 'rb') as file:
            P = dill.load(file)
    '''

    # finding gain k for Tracking :
    print('\n \n')
    print('======================== gain matrix ========================')
    print('generating gain_matrix using P')
    Psim = P[::-1]
    K_matrix = generate_gain_matrix(R, B_func, Psim, tvec, dynamic_symbs,
                                    traj_label)
    print('gain matrix is ready!')

    # saving K_matrix in a numpy file
    np.save('K_matrix' + '_' + label + '_' + traj_label + '_final_time_' +
            str(final_time) + '.npy', K_matrix)

    # ipydex.IPS()

    # finding states of the system using  K_matrix :

    xdot_func = sympy_states_to_func()
    print('\n \n')
    print('======================== x_closed_loop ========================')
    print('integrating to find x_closed_loop')

    # xa mit Abweichung !
    #xa = [0.0] + [np.pi/2
    #             for i in range(len(q) - 1)] + [0 for i in range(len(q))]

    xa = [0.0] + [np.pi
                  for i in range(len(q) - 1)] + [0.0 for i in range(len(q))]

    x_closed_loop = odeint(
        ode_function, xa, tvec, args=(xdot_func, K_matrix, tvec, traj_label))

    # x_open_loop= odeint(ode_function, xa, t, args=(xdot_func, K_matrix, t, 'Open_loop') )

    # saving x_closed_loop in a numpy file
    np.save('x_closed_loop' + '_' + label + '_' + traj_label + '.npy',
            x_closed_loop)

    np.save('u_closed_loop' + '_' + label + '_' + traj_label + '.npy',
            np.array(ct.tracking.tracking_res[traj_label]['ucl']))

    # returning the results :
    # ct.tracking.tracking_res[traj_label].update({
    #     'x_closed_loop': x_closed_loop
    # }
    # save_tracking_results_to_contianer(ct, traj_label,x_closed_loop)
        
    # print(ct.tracking.tracking_res[traj_label])
    # ct.tracking.tvec = tvec
    # ct.tracking.tracking_res[traj_label].update({'P': P})
    # ct.tracking.tracking_res[traj_label].update({'K_matrix': K_matrix})
    # ct.tracking.tracking_res[traj_label].update({
    #     'ucl':
    #     np.array(ct.tracking.tracking_res[traj_label]['ucl'])
    # })

    return