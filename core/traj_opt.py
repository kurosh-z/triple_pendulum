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
from pytrajectory import log

from pytrajectory import aux, TransitionProblem
log.console_handler.setLevel(10)

#=============================================================
# My Python modules
#=============================================================
from myfuncs import *
import ipydex

#=============================================================
# Trajectory Optimization
#=============================================================

ipydex.activate_ips_on_exception()


def trajectory_generator(ct, max_time, constraints=None):
    '''
    finding the trajectory using pytrajectory
    '''
    q = ct.model.q
    fx = ct.model.fx
    gx = ct.model.gx
    dynamic_symbs = ct.model.dynamic_symbs
    label = ct.label
    
    # converting sympy expressions to sympy functions to be used in pytrj_rhs

    ct.trajectory.qdd_functions = convert_qdd_to_func(fx, gx, dynamic_symbs)

    # senkrechte Anfangs- und Endbedingungen
    xa = [0.0] + [np.pi
                  for i in range(len(q) - 1)] + [0.0 for i in range(len(q))]
    xb = [0.0 for i in range(2 * len(q))]

    # ipydex.IPS()

    ua = [0.0]
    ub = [0.0]

    a = 0
    b= [2]
    # b = max_time- np.r_[0.25, 0.35, 0.45]
    seed=[19,29,390]
    # con = {0 : [-1.2, 1.2]}
    con={}

    print('ready to start trajectory optimiztion !')
    print('xa', xa)
    print('xb', xb)
    """

    if int(pytr.__version__.split(".")[1]) > 2:
        # zusätzliche Parameter, die notwendig sind, damit auch develop-Version konvergiert

        additional_parameters = \
        {
             "show_ir":  False,  # optionale grafische Anzeige von Zwischenergebnissen
             "use_std_approach": False,  # alte (unübliche Stützpunktdef.)
             "use_chains": False,  # Ausnutzung von Integratorketten deakt.
             "eps": 0.1,  # größere Fehlertoleranz (Endzustand)
             "maxIt": 5,
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
    
    
    
    # saving the results as numpy
    time_vec = np.linspace(0, max_time, 1000)
    x_traj = [cs_ret[0](t) for t in time_vec]
    u_traj = [[cs_ret[1](t) for t in time_vec]]

   
    
    np.save('x_traj' + '_' + label + '_'+ 'max_time_' + str(max_time) + '.npy',
            x_traj)
    np.save('u_traj' + '_' + label +  '_'+'max_time_' + str(max_time) + '.npy',
            u_traj)

    """

    
    # first_guess = {'seed': 25}
    # Parallelized :

    # S = TransitionProblem(
    #     pytraj_rhs,
    #     a,
    #     b,
    #     xa,
    #     xb,
    #     ua,
    #     ub,
    #     first_guess=first_guess,
    #     kx=2,
    #     eps=0.1,
    #     maxIt=6,
    #     use_chains=False,
    #     sol_steps=1300
    #     )

    args = aux.Container(
        poolsize=4,
        ff=pytraj_rhs,
        a=a,
        b=b,
        xa=xa,
        xb=xb,
        ua=0,
        ub=0,
        use_chains=False,
        ierr=None,
        maxIt=6,
        eps=0.3,
        kx=2,
        use_std_approach=False,
        seed=seed,
        constraints=con)




    results = aux.parallelizedTP(debug=False, save_results=False, **args.dict)

    seed_times=[(se, t) for t in b for se in seed]

    traj_labels = [
        '{}_{}'.format(seed_time[0], seed_time[1]) for seed_time in seed_times
    ]
    
    trajectories= zip(traj_labels,results)
    # solC = S.solve(return_format='info_container')
    # cont_dict = aux.containerize_splines(S.eqs.trajectories.splines)
    pfname = 'swingup_splines_' + label + '.pcl'
    with open(pfname, 'wb') as pfile:
        dill.dump(trajectories, pfile)
        print("Trajectories Written to {}".format(pfname))

    
    ct.trajectory.seed_times= seed_times
    ct.trajectory.pytrajectory_res= trajectories
    ct.trajectory.xa= xa
    ct.trajectory.xb= xb
    ct.trajectory.a= a
    ct.trajectory.b= b
    ct.trajectory.ua= ua
    ct.trajectory.ub= ub
    

    ipydex.IPS()
