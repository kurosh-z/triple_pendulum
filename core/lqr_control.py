# coding: utf-8
'''
-----------
To Do :
  - parameter shoul be read from a file isntead of dictionary!
  - finding a general way to define Q and R according to number of pendulums 
  - (should I have a dynamic way of changing of Q and R to find the best result ?! )
'''
#=============================================================
# Standard Python modules
#=============================================================

#=============================================================
# External Python modules
#=============================================================
#from __future__ import division, print_function
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
import mpmath as mp
import scipy as sc

#=============================================================
# Standard Python modules
#=============================================================
from functions import *
from sys_model import *

#=============================================================
# Lqr control for top equilibrium point
#=============================================================

# Defining equilibrium point and system parameters
equilib_point = sm.Matrix([0., 0., 0., 0.])

parameter_values = [(g, 9.81), (a[0], 0.2), (d[0], 10.0), (m[0], 3.34),
                    (m[1], 0.8512), (J[0], 0), (J[1], 0.01980), (f, 0)]

# linearization of model @ equilibrium point
A, B = linearize_state_equ(fx, gx, q, qdot, parameter_values, equilib_point)

#lqr control to find K (we need it as a boundry-value for our trajectory design !)
Q = np.identity(4)
R = np.identity(1)
k_top = lqr(A, B, Q, R)
#print(k_top)