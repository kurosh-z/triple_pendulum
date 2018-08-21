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
# from matplotlib_viz import pen_animation
from myfuncs import sympy_states_to_func
from myfuncs import load_sys_model
from myfuncs import load_traj_splines
from myfuncs import load_pytrajectory_results
from myfuncs import generate_sys_model_with_parameter_deviation
from myfuncs import parallelized_model_generator
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

tol_dicts = [{
    'l1': 0.01
}, {
    'l2': 0.01
}, {
    'l3': 0.01
}, {
    'a1': 0.01
}, {
    'a2': 0.01
}, {
    'a3': 0.01
}, {
    'm0': 0.01
}, {
    'm1': 0.01
}, {
    'm2': 0.01
}, {
    'm3': 0.01
}, {
    'J0': 0.01
}, {
    'J1': 0.01
}, {
    'J2': 0.01
}, {
    'J3': 0.01
}, {
    'd1': 0.01
}, {
    'd2': 0.01
}, {
    'd3': 0.01
}, {
    'g': 0.01
}]
cfg.pendata.model.tol_dicts= tol_dicts
# generate_sys_model_with_parameter_deviation(param_tol_dict=tol_dicts[0])
# parallelized_model_generator(tol_dicts)
# # # load_sys_model(cfg.pendata, model_without_param_deviation=True)
# load_sys_model(cfg.pendata, param_tol_dicts=tol_dicts)
# # # # generating trajectory with pytrajectory
# # # # trajectory_generator(cfg.pendata, max_time)


# # label = cfg.pendata.label
# # pfname = 'swingup_splines_' + label + '.pcl'
# # # load_traj_splines(cfg.pendata, pfname)

# # list of best splines to be loaded
# pfname1= 'swingup_splines__add_infos__39_1.9_splines_320_x0_None_x4_None_.pcl'
# pfname2= 'swingup_splines__add_infos__19_1.8_splines_320_x0_None_x4_None_.pcl'
# pfname3= 'swingup_splines__add_infos__19_1.7_splines_160_x0_None_x4_None_.pcl'
# pfname4= 'swingup_splines__add_infos__19_1.6_splines_320_x0_None_x4_None_.pcl'
# pfname5= 'swingup_splines__add_infos__29_2_splines_160_x0_None_x4_None_.pcl'
# pfnames=[pfname1, pfname2, pfname3, pfname4, pfname5]
# for pfname in pfnames :
#     load_pytrajectory_results(cfg.pendata, pfname)

# # tracking control of the time varying linear system
# parallelized_tracking_control(cfg.pendata, pool_size=4)

myplot(cfg.pendata)

# visualizing the results :
# visualization(cfg.pendata, mode='simulation', max_time=max_time)
# pen_animation(cfg.penda
# ta, filename='test')
