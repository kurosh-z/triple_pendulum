# coding: utf-8
'''
main function
Developer:
-----------
Kurosh Zamani

-----------
ToDo :



'''
#from __future__ import division, print_function
#=============================================================
# Standard Python modules
#=============================================================
import os, sys
import logging

#=============================================================
# External Python modules
#=============================================================

# from sympy.physics.vector import init_vprinting, vlatex
# init_vprinting(use_latex='mathjax', pretty_print=False)
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import cfg
from cfg import Pen_Container_initializer
from myfuncs import load_sys_model
from myfuncs import load_traj_splines
from myfuncs import load_pytrajectory_results
# from myfuncs import load_tracking_results
from myfuncs import load_traj_splines
from myfuncs import read_tracking_results_from_directories
from myfuncs import generate_model_names_form_tol_dict

import dill

import ipydex

#=============================================================
# myplot  :
# =============================================================


def myplot(ct):
    '''plots tracking results
    '''
    #     directories = [
    #         '137_1.5', '137_1.8', '19_1.6', '19_1.9', '39_1.7', '65_1.5', '65_1.8',
    #         '88_1.6', '88_1.9', '137_1.6', '137_1.9', '19_1.7', '39_1.5', '39_1.8',
    #         '65_1.6', '65_1.9', '88_1.7', '137_1.7', '19_1.5', '19_1.8', '39_1.6',
    #         '39_1.9', '65_1.7', '88_1.5', '88_1.8'
    #     ]

    #     directories = [
    #         '19_1.6', '19_1.9', '88_1.9', '137_1.9', '19_1.7', '19_1.5', '19_1.8',
    #         '39_1.9'
    #     ]
    # directories = ['19_1.9', '39_1.9']
    # directories = ['19_2', '29_2','390_2']
    directories = ['29_2']

    # load all the tracking results from directories in directories :
    tracking_dict = read_tracking_results_from_directories(directories)

    # load trajectorie results and convert to splines
    label = ct.label
    pfname = 'swingup_splines_' + label + '.pcl'
    load_traj_splines(ct, pfname)
    traj_dict = ct.trajectory.parallel_res
    print('trajectory resluts successfully loaded')

    frames_per_sec = 240
    final_time = 4
    tvec2 = np.linspace(0., final_time, frames_per_sec * final_time)

    figs = []

    deviation_list = ['parameter']
    tol_dicts=ct.model.tol_dicts
    model_types= generate_model_names_form_tol_dict(tol_dicts)[6:20]

    # model_types=['m3_0.01']
    xt_keys=['00']
    qr_keys= ['1.01.0']


    # deviation_percentage= '_0.01'

    idx=0
    for directory in directories:
        for deviation in deviation_list:
            for model_type in model_types:
                for xt_key in xt_keys:
                    for qr_key in qr_keys:

                        figs.append(plt.figure(figsize=(40, 30)))
                        grid = plt.GridSpec(4, 2, hspace=0.4, wspace=0.2)
                        # title of the figure :
                        plt.title(
                            u'Ergibnisse für : seed_time:{}, dev:{}, model_type:{}, boundry_dev: {}, QR : {}'.
                            format(directory, deviation, model_type, xt_key,
                                   qr_key))

                        # defining the postions of axex in a figure
                        ax_phis = figs[idx].add_subplot(grid[0:1, ::])
                        ax_phi_dot = figs[idx].add_subplot(grid[1:2, ::])
                        ax_x = figs[idx].add_subplot(grid[2:3, 0:-1])
                        ax_x_dot = figs[idx].add_subplot(grid[2:3, 1::])

                        ax_u = figs[idx].add_subplot(grid[3:4, 0:-1])
                        ax_k = figs[idx].add_subplot(grid[3:4, 1::])

                        tvec = tracking_dict[directory]['simulation_time']
                        k_matrix = tracking_dict[directory]['k_matrix'][deviation][model_type][xt_key][qr_key]
                        u_cl = tracking_dict[directory]['u_cl'][deviation][model_type][xt_key][qr_key]
                        x_cl = tracking_dict[directory]['x_cl'][deviation][model_type][xt_key][qr_key]

                        phis = x_cl[:, 1:4] * 180 / np.pi
                        phi_dots = x_cl[:, 5:]
                        x = x_cl[:, 0]
                        x_dot = x_cl[:, 4]
                        x_func_traj = traj_dict[directory][0]
                        x_traj = np.array([x_func_traj(t) for t in tvec2])

                        # ploting phis
                        ax_phis.plot(
                            tvec[::20], phis[:, 0][::20], 'ro', color='red', label='$q_1$')
                        ax_phis.plot(
                            tvec[::20],
                            phis[:, 1][::20],
                            'ro',
                            color='blue',
                            label='$q_2$')
                        ax_phis.plot(
                            tvec[::20],
                            phis[:, 2][::20],
                            'ro',
                            color='green',
                            label='$q_3$')

                        ax_phis.plot(
                            tvec2,
                            x_traj[:, 1] * 180 / np.pi,
                            color='red',
                            label='$q_1ref$')
                        ax_phis.plot(
                            tvec2,
                            x_traj[:, 2] * 180 / np.pi,
                            color='blue',
                            label='$q_2ref$')
                        ax_phis.plot(
                            tvec2,
                            x_traj[:, 3] * 180 / np.pi,
                            color='green',
                            label='$q_3ref$')

                        ax_phis.set_xlabel('Zeit (s)')
                        ax_phis.set_ylabel('Winkel (Grad)')
                        ax_phis.legend(loc=1)
                        ax_phis.grid(True)

                        # ploting phi_dots :
                        ax_phi_dot.plot(
                            tvec[::15], phi_dots[:, 0][::15], 'ro', label='$\dot{q_1}$')
                        ax_phi_dot.plot(
                            tvec[::15],
                            phi_dots[:, 1][::15],
                            'ro',
                            color='blue',
                            label='$\dot{q_2}$')
                        ax_phi_dot.plot(
                            tvec[::15],
                            phi_dots[:, 2][::15],
                            'ro',
                            color='green',
                            label='$\dot{q_3}$')

                        ax_phi_dot.plot(
                            tvec2, x_traj[:, 5], color='red', label='$\dot{q_1}_{ref}$')
                        ax_phi_dot.plot(
                            tvec2, x_traj[:, 6], color='blue', label='$\dot{q_2}_{ref}$')
                        ax_phi_dot.plot(
                            tvec2, x_traj[:, 7], color='green', label='$\dot{q_3}_{ref}$')

                        ax_phi_dot.set_xlabel('Zeit (s)')
                        ax_phi_dot.set_ylabel('Winkelgeschwindigkeit (rad/s)')
                        ax_phi_dot.legend(loc=4)
                        ax_phi_dot.grid(True)

                        #         # #ploting x :
                        ax_x.plot(tvec[::12], x[::12], 'ro', label='$q_0$')
                        ax_x.plot(tvec2, x_traj[:, 0], color='blue', label='$q_1ref$')

                        ax_x.set_xlabel('Zeit (s)')
                        ax_x.set_ylabel('Position des Wagens')
                        ax_x.legend(loc=1)
                        ax_x.grid(True)

                        #         # # ploting x_dot :
                        ax_x_dot.plot(tvec[::12], x_dot[::12], 'ro', label='$\dot{q_0}$')
                        ax_x_dot.plot(
                            tvec2, x_traj[:, 4], color='blue', label='$\dot{q_1}_{ref}$')

                        ax_x_dot.set_xlabel('Zeit (s)')
                        ax_x_dot.set_ylabel('Wagengeschwindigkeit (m/s)')
                        ax_x_dot.legend(loc=1)
                        ax_x_dot.grid(True)

                        #         # # ploting u :
                        ax_u.plot(
                            tvec, u_cl, 'k', tvec[::12], u_cl[::12], 'ro', label='$u_cl$')

                        ax_u.set_xlabel('Zeit s')
                        ax_u.set_ylabel('Input ($m/s^2$)')
                        ax_u.legend(loc=1)
                        ax_u.grid(True)
                        # #         # # ploting K_matrix
                        ax_k.plot(
                            tvec, k_matrix[:, 0], '--r', color='magenta', label='$k_0$')
                        ax_k.plot(tvec, k_matrix[:, 1], '--r', color='blue', label='$k_1$')
                        ax_k.plot(tvec, k_matrix[:, 2], '--r', color='cyan', label='$k_2$')
                        ax_k.plot(
                            tvec, k_matrix[:, 3], '--r', color='black', label='$k_3$')
                        ax_k.plot(
                            tvec, k_matrix[:, 4], '--r', color='green', label='$k_4$')
                        ax_k.plot(
                            tvec, k_matrix[:, 5], '--r', color='yellow', label='$k_5$')
                        ax_k.plot(tvec, k_matrix[:, 6], '--r', color='red', label='$k_6$')
                        ax_k.plot(
                            tvec, k_matrix[:, 7], '--r', color='brown', label='$k_7$')

                        ax_k.set_xlabel('Zeit (s)')
                        ax_k.set_ylabel('Folgereglerversarkung')
                        ax_k.legend(loc=2)
                        ax_k.grid(True)

                        # BASE_PATH='/home/kurosh/Documents/triple_pendulum/trackig_results'
                        # save_path= os.path.join(BASE_PATH, directory)
                        # os.chdir(save_path)
                        # fig_name= 'tracking_res_'+deviation + deviation_percentage+ '.png'
                        # figs[idx].savefig( fig_name)
                        # os.chdir('..')
                        idx+=1

    plt.show()
