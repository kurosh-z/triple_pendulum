
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
import os
import sys
import logging
#=============================================================
# External Python modules
#=============================================================
import numpy as np
from numpy import dot, arange, around
from scipy import sin, cos, pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import cfg
import ipydex


#=============================================================
# visualization  :
#=============================================================

def visualization(ct, filename=None):
    '''
    animating the resulsts 

    '''
    num_pen = ct.number_of_pendulums
    states = ct.tracking.x_closed_loop
    param_values = ct.parameter_values
    ref_traj = ct.trajectory.cs_ret[0]

    x_cart = states[:, 0]
    cart_width = 0.3
    cart_hight = 0.1

    rods_length = {}
    phi_pens = {}
    x_pens = {}
    y_pens = {}

    for i in range(num_pen):
        phi_pens.update({'xpen'+str(i+1): states[:, i]})
        rods_length.update({'rod'+str(i+1): param_values[i]})

        xpeni = rods_length['rod'+str(i+1)] * sin(phi_pens['xpen'+str(i+1)])
        x_pens.update({'x_pen'+str(i+1): xpeni})
        
