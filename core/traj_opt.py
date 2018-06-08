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
import dill
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
# from pytrajectory import log
# log.console_handler.setLevel(10)

#=============================================================
# My Python modules
#=============================================================
from functions import *
import ipydex

#=============================================================
# Trajectory Optimization
#=============================================================

# ipydex.activate_ips_on_exception()


def trajectory_optimization(ct, max_time, constraints=None):
    '''
    finding the trajectory
    '''
    q = ct.model.q
    fx = ct.model.fx
    gx = ct.model.gx
    dynamic_symbs = ct.model.dynamic_symbs
    # converting sympy expressions to sympy functions to b used in pytrj_rhs

    ct.trajectory.qdd_functions = convert_qdd_to_func(fx, gx, dynamic_symbs)

    # senkrechte Anfangs- und Endbedingungen
    xa = [0.0] + [np.pi
                  for i in range(len(q) - 1)] + [0.0 for i in range(len(q))]
    xb = [0.0 for i in range(2 * len(q))]

    # ipydex.IPS()

    ua = [0.0]
    ub = [0.0]

    print('ready to start trajectory optimiztion !')
    print('xa', xa)
    print('xb', xb)

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

    control_sys = pytr.ControlSystem(
        pytraj_rhs,
        a=0,
        b=max_time,
        xa=xa,
        xb=xb,
        ua=ua,
        ub=ub,
        **additional_parameters)

    cs_ret = control_sys.solve()

    if control_sys.reached_accuracy:
        print("Pytrajecotry Succeeded!")

    ct.trajectory.cs_ret = cs_ret
    ct.trajectory.pytraj_rhs = pytraj_rhs
    ct.trajectory.max_time= max_time
    ct.trajectory.xa= xa
    ct.trajectory.xb= xb
    '''
    with open('xs.pkl', 'wb') as file:
        dill.dump(config.cs_ret[0], file)
    '''
    # ipydex.IPS()