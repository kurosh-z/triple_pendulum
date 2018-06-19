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
# import os, sys
# import logging
#=============================================================
# External Python modules
#=============================================================

# from sympy.physics.vector import init_vprinting, vlatex
# init_vprinting(use_latex='mathjax', pretty_print=False)
# import ipydex

import cfg
from cfg import Pen_Container_initializer
from sys_model import system_model_generator
# from traj_opt import trajectory_optimization
# from tracking_control import tracking_control
# from pydy_viz import visualization
# from matplotlib_viz import pen_animation
from functions import sympy_states_to_func

#=============================================================
# main  :
# =============================================================
number_of_pendulums = 1
mode = 'matlab'
max_time = 2


Pen_Container_initializer(number_of_pendulums)

# modelig the system with kanes' Method
system_model_generator(cfg.pendata)
cfg.pendata.system_sate_func = sympy_states_to_func()

def myfunc1(x):
    states=x
    u=10.0
    # ret = cfg.pendata.system_sate_func(states, u)
    ret= u + states
    return ret

def search(words):
    """Return list of words containing 'son'"""
    newlist = [w for w in words if 'son' in w]
    return newlist
