# coding: utf-8
'''
main function 
Developer:
-----------
Kurosh Zamani

-----------
ToDo :
  


'''
#from __future__ import deviation, print_function
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
from tracking_control import parallelized_tracking_control
from pydy_viz import visualization
from matplotlib_viz import pen_animation
from myfuncs import sympy_states_to_func
from myfuncs import load_sys_model
from myfuncs import load_traj_splines
from myfuncs import load_pytrajectory_results
from myfuncs import generate_sys_model_with_parameter_deviation
from myplots import myplot

# from traj_DRICON import trajectory_generator
# from trajq0_generation import trajectory_generator
import ipydex
#=============================================================
# main  :
# =============================================================
number_of_pendulums = 3
max_time = 2

# initializing the container. it makes a golobal variable
# pendata to which is an instance of Pen_container Class
Pen_Container_initializer(number_of_pendulums)
# modeling the system with kanes' Method
# system_model_generator(cfg.pendata)
generate_sys_model_with_parameter_deviation(cfg.pendata,default_tol=0.03)

# load_sys_model(cfg.pendata)
# load_sys_model(cfg.pendata, model_with_deviation=True)

# generating trajectory with pytrajectory
# trajectory_generator(cfg.pendata, max_time)

# label = cfg.pendata.label
# pfname = 'swingup_splines_' + label + '.pcl'
# # load_traj_splines(cfg.pendata, pfname)
# load_pytrajectory_results(cfg.pendata, pfname)

# # tracking control of the time varying linear system
# parallelized_tracking_control(cfg.pendata, pool_size=2)

# # myplot(cfg.pendata)
# # ipydex.IPS()

# visualizing the results :
# visualization(cfg.pendata, mode='simulation', max_time=max_time)
# pen_animation(cfg.penda
# ta, filename='test')
