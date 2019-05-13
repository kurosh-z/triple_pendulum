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

import cfg
import numpy as np
from cfg import Pen_Container_initializer
# from sys_model import system_model_generator
from myfuncs import system_model_generator_original
from myfuncs import generate_sys_model_with_parameter_deviation
from myfuncs import parallelized_model_generator
from myfuncs import load_sys_model
from myfuncs import sympy_states_to_func
# from trajOpt2 import TrajProblem
from tracking_control import parallelized_tracking_control
from tracking_control import _tracking_control
from myplots import myplot

from trajOpt import TrajOptimization
from myfuncs import load_traj_from_numpyFile
import ipydex
ipydex.activate_ips_on_exception()
#=============================================================
# main  :
# =============================================================
number_of_pendulums = 3

# initializing the container. it makes a golobal variable
Pen_Container_initializer(number_of_pendulums)
ct= cfg.pendata
# example of making parameter deviation dictionary to generate the models desired:
# for param_dict in  ct.parameter_tol_dicts:
#     for key in param_dict:
#         param_dict[key]*=(-.2)

# parallelized_model_generator(ct.parameter_tol_dicts) 

generate_sys_model_with_parameter_deviation(default_tol= 0.03) # use this instead of param_dict if you just want the same percentage of deviation (default_tol) for every parameters

# system_model_generator_original(ct)
# load_sys_model(ct, model_without_param_deviation=True)
# load_sys_model(ct, param_tol_dicts=ct.parameter_tol_dicts)

ff = sympy_states_to_func(ct.model.fx, ct.model.gx)


def dynamics(tt, xx, uu):
    '''
    '''
    xx_dot=[]
    for x, u in zip(xx, uu):
        xx_dot.append(ff(x, u).tolist())

    return np.array(xx_dot)

############################################
# using data from Matlab :

traj= TrajOptimization(dynamics=dynamics)
xFunc, uFunc= traj.convertGridsToFunc('Mat50_2.2.mat')
traj_keys=['mat50_2.2']  # list of trajecotries we want to use for tracking
ct.trajectory.parallel_res={traj_keys[0]: (xFunc, uFunc)}


##S# save functions to be used in tracking :

## example of tracking for just one model ( original model wihtout deviations)
# traj_key= 'mat50_2.2'
# arg = dict([('traj_key', traj_key), ('deviation', 'None')])
# _tracking_control(arg)

## example of tracking control using paralleleized tracking :
parallelized_tracking_control(ct, pool_size=4)


# plotting the results :
myplot(ct)




###########################################
# solve problem with python :

# def pathObj(tt, xx, uu):

#     obj=0.0
#     for u in uu :
#         obj+=u**2
#     return obj


# guess = {
#     'finalTime': 2,
#     'initialState': [0, np.pi, np.pi, np.pi, 0, 0, 0, 0],
#     'finalState': [0, 0, 0, 0, 0, 0, 0, 0],
#     'initialControl': [0],
#     'finalControl': [0]
# }

# eps=1e-4
# PI= np.pi
# inf=np.inf
# bound={
#     'initialState': {'Low':(np.array(guess['initialState'])- eps).tolist() , \
#                      'Upp' : (np.array(guess['initialState'])+ eps).tolist()}, \
#     'finalState': {'Low':(np.array(guess['finalState'])- eps).tolist() , \
#                    'Upp' : (np.array(guess['finalState'])+ eps).tolist()}, \
#     'state': {'Low': [-2, -2*PI, -2*PI, -2*PI, -inf, -inf, -inf, -inf ], \
#               'Upp': [ 2,  2*PI,  2*PI,  2*PI, inf, inf, inf, inf ]},

#     'initialControl':{'Low':[0-eps] , \
#                       'Upp':[0+eps]},

#     'finalControl':{'Low':[0-eps] , \
#                     'Upp':[0+eps]}

# }

# prob = TrajProblem(guess=guess,bounds=bound, dynamics=dynamics, pathObjective=pathObj)

# prob.solve(nSegment=8)

# xFunc, uFunc= traj.convertGridsToFunc()

# # save functions to be used in tracking :
# traj_key= 'py20_2.1'
# ct.trajectory.parallel_res={traj_key: (xFunc, uFunc)}



# ipydex.IPS()
