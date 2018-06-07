'''
main function 
Developer:
-----------
Kurosh Zamani

-----------
To Do :
  


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
import ipydex

import cfg
from cfg import Pen_Container_initializer
from sys_model import system_model_generator
from traj_opt import trajectory_optimization
from tracking_control import tracking_control


#=============================================================
# main  :
#=============================================================

# initializing the container to pass to the functions 
#  as global variable :
Pen_Container_initializer(1)

# modeling the system with kanes' Method
system_model_generator(cfg.pendata)

# generating trajectory with pytrajectory
max_time= 2
trajectory_optimization(cfg.pendata, max_time)

# tracking control of the time varying linear system
tracking_control(cfg.pendata)

