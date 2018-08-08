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

import dill
#=============================================================
# External Python modules
#=============================================================

# from sympy.physics.vector import init_vprinting, vlatex
# init_vprinting(use_latex='mathjax', pretty_print=False)
# import ipydex

import cfg
from cfg import Pen_Container_initializer
from sys_model import system_model_generator
from traj_opt import trajectory_generator
from tracking_control import tracking_control
from pydy_viz import visualization
from matplotlib_viz import pen_animation
from myfuncs import sympy_states_to_func
# from traj_DRICON import trajectory_generator
<<<<<<< HEAD
# from traj_generation import trajectory_generator
from traj_opt import  trajectory_optimization

#=============================================================
# main  :
# =============================================================
number_of_pendulums = 1
mode = 'load'
=======
# from trajq0_generation import trajectory_generator
import ipydex
#=============================================================
# main  :
# =============================================================
number_of_pendulums = 3
mode = 'simulation'
>>>>>>> 591b3ec216c0347453eefd7f8fdc373746f9633d
max_time = 2

if mode == 'load':

    Pen_Container_initializer(number_of_pendulums)
    # modelig the system with kanes' Method
    system_model_generator(cfg.pendata)
    # visualizing the  results from pas simulations  :

    visualization(cfg.pendata, mode=mode, max_time=max_time)

else:
    # initializing the container. it makes a golobal variable
    # pendata to which is an instance of Pen_container Class
    Pen_Container_initializer(number_of_pendulums)

    # modeling the system with kanes' Method
<<<<<<< HEAD
    system_model_generator(cfg.pendata)
    

    # generating trajectory with pytrajectory
    '''
    trajectory_optimization(cfg.pendata, max_time)
    # trajectory_generator(cfg.pendata, max_time)

    # tracking control of the time varying linear system
=======
    #system_model_generator(cfg.pendata)
    label=cfg.pendata.label
    with open('sys_model_' + label+ '.pkl', 'rb') as file :

        model=dill.load(file)
    
    ipydex.IPS()

    # generating trajectory with pytrajectory

    #  trajectory_optimization(cfg.pendata, max_time)
    # trajectory_generator(cfg.pendata, max_time)

    # # tracking control of the time varying linear system
>>>>>>> 591b3ec216c0347453eefd7f8fdc373746f9633d
    # tracking_control(cfg.pendata)

    # visualizing the results :
    # visualization(cfg.pendata, mode='simulation', max_time=max_time)
    # pen_animation(cfg.pendata, filename='test')

    '''