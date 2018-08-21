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


def parallelized_tracking_control(ct, pool_size=4):
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


    # constructing trackt_dict :

    traj_multiple_args= []
    model_names= ct.model.models_with_tol_dicts['model_list']

    for traj_key in traj_keys:
        for model_type_label in model_names :
            arg= dict([('traj_key', traj_key), ('deviation','parameter'), ('model_type_label',('model_with_tol_dict', model_type_label) )])
            traj_multiple_args.append(arg)


    processor_pool = Pool(pool_size)
    processor_pool.map(_tracking_control , traj_multiple_args )


    # Process_jobs=[]
    # for arg in  (traj_multiple_args[0], ):
    #     p= multiprocessing.Process(target= _tracking_control, args=(arg,))
    #     Process_jobs.append(p)
    #     p.start()
    #     p.join()


    return


def _tracking_control(track_dict):
    '''
    track_dict : a dictionar with folowing keys :
    

    'traj_key' : trajectory key name

    'deviation' : could be one of these key words :
              parameter : tracking with model parameter divison
              boundry_deviation : tracking with boundry_value deviation
              both= parameter and boundry_value deviations
              None= with no deviation  

    'model_type_label' : could be one of folowing key words
     ***(you dont need this without parameter deviation) :
                 ('model_with_default_tol', model_name) 
                 ('model_with_tol_dict', model_name)    

    'boundry_deviation' : (t0, deviation percentage) :
     *** (you dont need this wihtout boundry deviation)

     'QR_factor' : optional : tupel (Q_factor, R_factor)
    '''
    ct= cfg.pendata
    traj_label= track_dict['traj_key']
    deviation= track_dict['deviation']

    if (deviation == 'None' or deviation == 'boundry_deviation'):
        fx= ct.model.fx
        gx= ct.model.gx
        model_key= 'original'
        model_type= 'original'

    if (deviation  == 'parameter' or deviation == 'both') :
        model_type= track_dict['model_type_label'][0]
        model_key = track_dict['model_type_label'][1]

        if model_type == 'model_with_default_tol':
            fx=ct.model.models_with_default_tol[model_key]['fx']
            gx=ct.model.models_with_default_tol[model_key]['gx']
            print('fx for model_with_defualt_tol is loaded ')

        elif model_type == 'model_with_tol_dict':
            fx= ct.model.models_with_tol_dicts[model_key]['fx']
            gx= ct.model.models_with_tol_dicts[model_key]['gx']
            print('fx and gx for model with tol dict are loaded')

    if (deviation == 'boundry_deviation' or deviation == 'both' ):
        boundry_deviation = track_dict['boundry_deviation']
        t0= boundry_deviation[0]
        x0_deviation = boundry_deviation[1]
    else :
        t0=0
        x0_deviation=0

    # fiding other values
    dynamic_symbs = ct.model.dynamic_symbs
    q = ct.model.q
    u = ct.model.q
    final_time = float(traj_label.split("_")[1])
    label = ct.label


    print('{} - {} : tracking started !'.format(traj_label, model_key))




    # defining equilibrium point for system linearization at top
    x0 = [0.0 for i in range(2 * len(q))]
    u0 = [0.0]
    equilibrium_point = x0 + u0

    # linearization of model @ equilibrium point
    A, B = linearize_state_equ(fx, gx, dynamic_symbs, equilibrium_point)
    print('{} - {} : linearized model is ready !'.format(traj_label, model_key))

    # lqr control to find K (we need it as a boundry-value (Pe) for our tracking !)
    Q_factor=1.0
    R_factor=1.0
    Q = Q_factor *np.identity(2 * len(q))
    R = R_factor * np.identity(1)

    # k_top = lqr(A, B, Q, R)
    results = lqr(A, B, Q, R, additional_outputs=True)
    k_top = results[0]

    # reshape P matrix and use it as initial guess for P
    Pe = results[1].reshape((2 * len(q))**2)

    # finding linearized model as a function of equilibrium point
    A_func, B_func = linearize_state_equ(
        fx, gx, dynamic_symbs, output_mode='numpy_func')



    # convert containerized results of pytrajectory to functions!
    convert_container_res_to_splines(ct, traj_label, model_key)

    # solving riccati differential equations inverse in time :
    frames_per_sec = 240
    tvec = np.linspace(0.0, final_time, final_time * frames_per_sec)

    print('{} - {} : Integrating riccati differential equations '.format(traj_label, model_key))


    P = odeint(
        riccati_diff_equ,
        Pe,
        tvec[::-1],
        args=(A_func, B_func, Q, R, dynamic_symbs, traj_label))


    # with open('P_matrix_' + label + traj_label + '.pkl', 'wb') as file:
    #     dill.dump(P, file)
    '''
    with open('P_matrix_'+ label+traj_label+'.pkl', 'rb') as file:
            P = dill.load(file)
    '''

    # finding gain k for Tracking :
    print('{} - {} : generating gain_matrix for '.format(traj_label, model_key))
    Psim = P[::-1]
    K_matrix = generate_gain_matrix(R, B_func, Psim, tvec, dynamic_symbs,
                                    traj_label)
    print('{} - {} : gain matrix is ready!'.format(traj_label, model_key))

    # np.save('K_matrix' + '_' + label + '_' + traj_label+ '.npy', K_matrix)


    # extend K-matrix and tvec for another 2 seconds :
    step=tvec[1]-tvec[0]
    extend_time= [tvec[-1]+i*step for i in range(1, 2*frames_per_sec)]
    sim_time= np.array(tvec.tolist() + extend_time)
    np_save(traj_label, 'simulation_time', sim_time)

    extend_k=[K_matrix[-1].tolist() for i in range(1, 2*frames_per_sec) ]
    K_matrix_extended= np.array(K_matrix.tolist()+ extend_k)

    # saving K_matrix in a numpy file
    file_name_k = 'K_matrix' +'_deviation_' + deviation + '_' + model_key + '_t0_' + str(t0)+'_x0_'+ str(x0_deviation) + '_QR_'+str(Q_factor)+str(R_factor)+'_'+ label + '_' + traj_label+ '.npy'
    np_save(traj_label,file_name_k, K_matrix_extended)


    # converting sympy to funcs :
    xdot_func = sympy_states_to_func(fx, gx)

    print('{} - {} : integrating to find x_closed_loop '.format(traj_label, model_key))

    # check if we should consider deviation in boundry conditions

    if (deviation == 'boundry_deviation' or deviation == 'both' ):
        x_on_traj= ct.trajectory.parallel_res[traj_label][0](t0)
        xa= x0_deviation * x_on_traj

        # find the indx of array sim_time at whitch approximatly sim_time[idx] = t0
        temp= np.absolute(sim_time-t0)
        ind= np.unravel_index(np.argmin(temp, axis=None), temp.shape)[0]
        new_sim_time= sim_time[ind:]
        
        print('t0',t0)
        print('x0', x0)
        print('x0_deviation',x0_deviation)
        print('x_on_traj',x_on_traj)
        
        # find x_closed_loop for desired deviation at desired time :
        x_closed_loop = odeint(
            ode_function,
            xa,
            new_sim_time,
            args=(xdot_func, K_matrix_extended, sim_time, traj_label))        
        
        # form 0 to t0 add 0 to x_closed_loop (just to have a uniform results
        # and in ploting it stresses the fact that we consider t0 as start point !) 
        x_closed_loop = [0 for t in sim_time[:ind]] + [x_closed_loop[i] for i, t in enumerate(new_sim_time)]



    else:
        xa = [0.0] + [np.pi for i in range(len(q) - 1)
                      ] + [0.0 for i in range(len(q))]

        x_closed_loop = odeint(
            ode_function,
            xa,
            sim_time,
            args=(xdot_func, K_matrix_extended, sim_time, traj_label))





    # generate u_closed_loop
    u_closed_loop= calculate_u_cl(ct,x_closed_loop,K_matrix_extended, sim_time, traj_label)
    # u_closed_loop= ct.tracking.tracking_res[traj_label]['ucl']

    # x_open_loop= odeint(ode_function, xa, t, args=(xdot_func, K_matrix, t, 'Open_loop') )

    # saving x_closed_loop in a numpy file
    x_cl_file_name='x_closed_loop' + '_deviation_' + deviation +  '_' + model_key + '_t0_' + str(t0)+'_x0_'+ str(x0_deviation) + '_QR_'+str(Q_factor)+str(R_factor) + '_'+ label + '_' + traj_label + '.npy'
    u_cl_file_name='u_closed_loop' + '_deviation_' + deviation +  '_' + model_key + '_t0_' + str(t0)+'_x0_'+ str(x0_deviation) + '_QR_'+str(Q_factor)+str(R_factor) + '_'+label + '_' + traj_label + '.npy'

    np_save(traj_label, x_cl_file_name, x_closed_loop)
    np_save(traj_label, u_cl_file_name, u_closed_loop )

    print('{} - {} : tracking  finished result are stored!'.format(traj_label, model_key))
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