# coding: utf-8
'''
-----------
To Do :
  - resolve the issues with imports 
  - praramer should be read form a file (or maybe using config.py 
    would be more appropriate hier ?!)
  
'''
#=============================================================
# Standard Python modules
#=============================================================
import sys, os
# import dill as pickle
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
import mpmath as mp
import scipy as sc
from sympy import latex

import symbtools as st
import symbtools.modeltools as mt
import symbtools.noncommutativetools as nct
from scipy.integrate import odeint
import pytrajectory as pytr
from pytrajectory import log
# log.console_handler.setLevel(10)

#=============================================================
# My Python modules
#=============================================================
#from functions import *
from sys_model import *
import config
import ipydex

#=============================================================
# Trajectory Optimization
#=============================================================

ipydex.activate_ips_on_exception()

# defining system parameters
#TO DO : system parameters should be read from a file !
# parameter_values = [(g, 9.81), (a[0], 0.2), (d[0], 0.010), (m[0], 3.34),
#                     (m[1], 0.8512), (J[0], 0), (J[1], 0.01980), (f, 0)]
# param_dict = dict(parameter_values)

# converting sympy expressions to functions to b used in pytrj_rhs

config.qdd_functions = convert_qdd_to_func(fx, gx, q, qdot, u, param_dict)

# senkrechte Anfangs- und Endbedingungen
xa = [0.0] + [np.pi for i in range(len(q) - 1)] + [0.0 for i in range(len(q))]
xb = [0.0 for i in range(2 * len(q))]
ipydex.IPS()
ua = [0.0]
ub = [0.0]

print('ready to start trajectory optimiztion !  \nxa:\nxb:\n ', xa, xb)
if int(pytr.__version__.split(".")[1]) > 2:
    # zusätzliche Parameter, die notwendig sind, damit auch develop-Version konvergiert

    additional_parameters = \
    {
         "show_ir":  False,  # optionale grafische Anzeige von Zwischenergebnissen
         "use_std_approach": False,  # alte (unübliche Stützpunktdef.)
         "use_chains": False,  # Ausnutzung von Integratorketten deakt.
         "eps": 0.05,  # größere Fehlertoleranz (Endzustand)
         "ierr": None,  # Intervallfehler ignorieren
     }
else:
    additional_parameters = \
    {
        "use_std_approach": False,  # alte (unübliche Stützpunktdef.)
        "use_chains": False,  # Ausnutzung von Integratorketten
     }

cs = control_sys = pytr.ControlSystem(
    pytraj_rhs,
    a=0,
    b=2.0,
    xa=xa,
    xb=xb,
    ua=ua,
    ub=ub,
    **additional_parameters)
config.cs_ret = cs.solve()

if cs.reached_accuracy:
    print("Pytrajecotry Succeeded!")

# with open('xs.pkl', 'wb') as file:
#     pickle.dump(cs_ret[0], file)

ipydex.IPS()