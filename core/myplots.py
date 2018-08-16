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
from myfuncs import load_tracking_results
from myfuncs import load_traj_splines

import dill

import ipydex


#=============================================================
# myplot  :
# =============================================================

def myplot(ct):
    '''plots tracking results
    '''
    file_names=['x_closed_loop_Inverted_simple_Pendulum_29_1.8.npy',
    'u_closed_loop_Inverted_simple_Pendulum_55_1.9.npy',
    'x_closed_loop_Inverted_simple_Pendulum_55_1.9.npy',
    'u_closed_loop_Inverted_simple_Pendulum_29_1.8.npy'
    ]

    load_tracking_results(ct, file_names)
    tracking_dict= ct.tracking.tracking_res

    label = ct.label
    pfname = 'swingup_splines_' + label + '.pcl'
    load_traj_splines(ct, pfname)
    traj_dict= ct.trajectory.parallel_res

    
    
    keys, values= zip(*ct.tracking.tracking_res.items())
    l= len(keys)*4

    

    frames_per_sec = 120
    final_time = 3
    tvec = np.linspace(0.0, final_time, final_time * frames_per_sec)
    tvec2= np.linspace(0., final_time, 1000)
    
    figs= []
    # axes= dict()
    const= 6
    i=0
    key= keys[i]
    

    for idx, key in enumerate(keys) :

        figs.append(plt.figure(figsize=(30, 20)))
        grid= plt.GridSpec(2,4, hspace=0.2, wspace=0.3)
        
        # defining the postions of axex in a figure
        ax_phis = figs[idx].add_subplot(grid[0:1, :-1])
        ax_phi_dot= figs[idx].add_subplot(grid[1:2,:-1])
        ax_x = figs[idx].add_subplot(grid[0:1, 1:])
        ax_x_dot= figs[idx].add_subplot(grid[1:2, 1:])
        ax_u= figs[idx].add_subplot(grid[2:3, :])
        ax_k= figs[idx].add_subplot(grid[-1, :])
        
        k_matrix= tracking_dict[key]['K_matrix']
        u_cl= tracking_dict[key]['u_closed_loop']
        x_cl= tracking_dict[key]['x_closed_loop']
        phis= x_cl[:,1:4]* 180/ np.pi
        phi_dots= x_cl[:,5:]
        x= x_cl[:, 0]
        x_dot= x_cl[:, 4]
        x_func_traj= traj_dict[key][0]
        
        # title of the figure :
        plt.title('Ergibnisse für'+ str(key))
        # ploting phis            
        ax_phis.plot(tvec[::3], phis[:,1][::3], 'ro',color='blue', label='$q_2$'  )
        ax_phis.plot(tvec[::3], phis[:,0][::3], 'ro', label='$q_1$' )
        ax_phis.plot(tvec[::3], phis[:,2][::3], 'ro',color='green', label='$q_3$'  )

        ax_phis.plot(tvec2, x_func_traj(tvec2)[:,1]*180/np.pi, color='red',label='$q_1ref$')
        ax_phis.plot(tvec2, x_func_traj(tvec2)[:,2]*180/np.pi, color='blue',label='$q_2ref$')
        ax_phis.plot(tvec2, x_func_traj(tvec2)[:,3]*180/np.pi, color='green',label='$q_3ref$')

        ax_phis.set_xlabel('Zeit (s)')
        ax_phis.set_ylabel('Winkel (Grad)')
        ax_phis.legend(loc=1)
        ax_phis.grid(True)
        
        # ploting phi_dots :
        ax_phi_dot.plot(tvec[::3], phi_dots[:,0][::3], 'ro', label='$\dot{q_1}$' )
        ax_phi_dot.plot(tvec[::3], phi_dots[:,1][::3], 'ro',color='blue', label='$\dot{q_2}$'  )
        ax_phi_dot.plot(tvec[::3], phi_dots[:,2][::3], 'ro',color='green', label='$\dot{q_3}$'  )

        ax_phi_dot.plot(tvec2, x_func_traj(tvec2)[:,5], color='red',label='$\dot{q_1}ref$')
        ax_phi_dot.plot(tvec2, x_func_traj(tvec2)[:,6], color='blue',label='$\dot{q_2}ref$')
        ax_phi_dot.plot(tvec2, x_func_traj(tvec2)[:,7], color='green',label='$\dot{q_3}ref$')
        
        ax_phi_dot.set_xlabel('Zeit (s)')
        ax_phi_dot.set_ylabel('Winkelgeschwindigkeit (rad/s)')
        ax_phi_dot.legend(loc=4)
        ax_phi_dot.grid(True)


        #ploting x :
        ax_x.plot(tvec[::3], x[::3], 'ro', label='$q_0$' )
        ax_x.plot(tvec2, x_func_traj(tvec2)[:,0], color='blue',label='$q_1ref$')
        
        ax_x.set_xlabel('Zeit (s)')
        ax_x.set_ylabel('Position des Wagens')
        ax_x.legend(loc=1)
        ax_x.grid(True)

        # ploting x_dot :
        ax_x_dot.plot(tvec[::3], x_dot[::3], 'ro', label='$q_0$' )
        ax_x_dot.plot(tvec2, x_func_traj(tvec2)[:,4], color='blue',label='$q_1ref$')
        
        ax_x_dot.set_xlabel('Zeit (s)')
        ax_x_dot.set_ylabel('Geschwindigkeit des Wagens (m/s)')
        ax_x_dot.legend(loc=1)
        ax_x_dot.grid(True) 

        # ploting u :
        ax_u.plot(tvec[::3], u_cl[::3], 'ro', label='$u_cl$')

        ax_u.set_xlabel('Zeit s')
        ax_u.set_ylabel('Input ($m/s^2$)')
        ax_u.legend(loc=1)
        ax_u.grid(True) 

        # ploting K_matrix
        ax_k.plot(tvec, k_matrix[:,0], '--r', color= 'magenta', label='$k_0$' )
        ax_k.plot(tvec, k_matrix[:,1], '--r',color='blue', label='$k_1$'  )
        ax_k.plot(tvec, k_matrix[:,2], '--r',color='cyan', label='$k_2$' )
        ax_k.plot(tvec, k_matrix[:,3], '--r',color='black', label='$k_3$' )
        ax_k.plot(tvec, k_matrix[:,4], '--r',color='green', label='$k_4$' )
        ax_k.plot(tvec, k_matrix[:,5], '--r',color='yellow', label='$k_5$' )
        ax_k.plot(tvec, k_matrix[:,6], '--r', color='red', label='$k_6$' )
        ax_k.plot(tvec, k_matrix[:,7], '--r', color='brown', label='$k_7$' )
        
        
        ax_k.set_xlabel('Zeit (s)')
        ax_k.set_ylabel('Folgereglerversarkung')
        ax_k.legend(loc=2)
        ax_k.grid(True)


    plt.show()     


    
    return