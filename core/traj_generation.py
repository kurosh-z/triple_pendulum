'''
main function 
Developer:
-----------
Kurosh Zamani

-----------
TODO :
  


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
# import ipydex
import numpy as np
from scipy.optimize import minimize

import cfg
from sys_model import system_model_generator
from functions import sympy_states_to_func
from functions import jac_func

import ipydex

ipydex.activate_ips_on_exception()

def interpolation_constrains(z):
    '''return collocation constrains
    '''
     # k :number of knotes ,  and n :system dimension
    k = cfg.pendata.trajectory.k
    n = cfg.pendata.trajectory.n
    max_time = cfg.pendata.trajectory.max_time
    hk = max_time / (k - 1)
    fxu = cfg.pendata.trajectory.fxu

    x = z[0:(2 * k - 1) * n].reshape(2*k-1, n)
    x_knot= x[::2]
    xc = x[1::2]

    # u_knote: uk on each knote , uc: u(k+1/2) 
    u_knot = z[(2 * k - 1) * n::2]
    uc = z[(2 * k - 1) * n + 1::2]

    coll_const = []
    inter_const = []
    for i in range(k - 1):

        x0 = x_knot[i]
        x1 = x_knot[i + 1]
        x01 = xc[i]

        u0 = [u_knot[i]]
        u1 = [u_knot[i + 1]]
        u01 = [uc[i]]

        f0 = fxu(x0, u0)
        f1 = fxu(x1, u1)
        # f01 = fxu(x01, u01)

        # coll_const.append(x0 - x1 - hk / 6.0 * (f0 + 4 * f01 + f1))
        inter_const.append(0.5 * (x0 + x1) + hk / 8.0 * (f0 - f1) - x01)
 
  
    return np.array(inter_const).ravel()

def collocation_constrains(z):
    '''return collocation constrains
    '''
    # k :number of knotes ,  and n :system dimension
    k = cfg.pendata.trajectory.k
    n = cfg.pendata.trajectory.n
    max_time = cfg.pendata.trajectory.max_time
    hk = max_time / (k - 1)
    fxu = cfg.pendata.trajectory.fxu

    x = z[0:(2 * k - 1) * n].reshape(2*k-1, n)
    x_knot= x[::2]
    xc = x[1::2]

    # u_knote: uk on each knote , uc: u(k+1/2) 
    u_knot = z[(2 * k - 1) * n::2]
    uc = z[(2 * k - 1) * n + 1::2]

    coll_const = []
    # inter_const = []
    for i in range(k - 1):

        x0 = x_knot[i]
        x1 = x_knot[i + 1]
        x01 = xc[i]

        u0 = [u_knot[i]]
        u1 = [u_knot[i + 1]]
        u01 = [uc[i]]

        f0 = fxu(x0, u0)
        f1 = fxu(x1, u1)
        f01 = fxu(x01, u01)

        coll_const.append(x0 - x1 - hk / 6.0 * (f0 + 4 * f01 + f1))
        # inter_const.append(0.5 * (x0 + x1) + hk / 8.0 * (f0 - f1) - x01)
 
  
    return np.array(coll_const).ravel()

def objective_functional(z):
    '''returns objective function Ju 
    '''
    k = cfg.pendata.trajectory.k
    n = cfg.pendata.trajectory.n
    max_time = cfg.pendata.trajectory.max_time
    hk = max_time / (k - 1)

    x = z[0:(2 * k - 1) * n].reshape(2*k-1, n)
    x_knot= x[::2]
    xc = x[1::2]

    # u_knote: uk on each knote , uc: u(k+1/2) 
    u_knot = z[(2 * k - 1) * n::2]
    uc = z[(2 * k - 1) * n + 1::2]

    xf=x[-1]

    S = Q = np.eye(4)
    R = 0.01 * np.identity(1)
    Ju = 0.5 * xf.dot(S).dot(xf)

    # defining l(x,u, t)
    lxut = lambda x, u: 0.5 * x.dot(Q).dot(x) + 0.5 * u.dot(R).dot(u)

    for i in range(k-1):
        
        x0 = x_knot[i]
        x1 = x_knot[i + 1]
        x01 = xc[i]

        u0 = np.array([u_knot[i]])
        u1 = np.array([u_knot[i + 1]])
        u01 = np.array([uc[i]]) 
        
        # Updating our J(u) 
        Ju += hk / 6.0 * (lxut(x0, u0) + 4 * lxut(x01, u01) + lxut(x1, u1))

    return Ju 


def trajectory_generator(ct, max_time):
    ''' generates trajectory using collocation method
    '''
    print('\n \n')
    print('======================== Trajectory_generator ========================')

    q = ct.model.q
    ct.trajectory.n= n = 2*len(q)
    ct.trajectory.k= k = 20

    ct.trajectory.max_time= h = max_time  

    x0 = [0.0] + [np.pi
                  for i in range(len(q) - 1)] + [0.0 for i in range(len(q))]
    xf = [0.0 for i in range(2 * len(q))]

    
    u0 = [0.0]
    uf = [0.0]

    # fxu needet for constraint generator
    fxu = sympy_states_to_func()
    ct.trajectory.fxu = fxu

    # defining constraints
    collocation_cons = {'type': 'eq', 'fun': collocation_constrains}
    interpolation_cons = {'type': 'eq', 'fun': interpolation_constrains}


    # boundry conditions
    boundry_x0 = {'type': 'eq', 'fun': lambda z: z[:n] - np.array(x0)}
    boundry_u0 = {'type': 'eq', 'fun': lambda z: z[(2*k-1) * n] - np.array(u0)}
    boundry_xf = {'type': 'eq', 'fun': lambda z: z[(2*k-1)*n - n : (2*k-1)*n] - np.array(xf)}
    boundry_uf = {'type': 'eq', 'fun': lambda z: z[-1] - np.array(uf)}

    # all constrainsts together
    cons = (collocation_cons,interpolation_cons, boundry_x0, boundry_xf)
    #cons = (collocation_cons, interpolation_cons,boundry_x0, boundry_xf)
    
    bnds= [(None, None) for i in range((2*k-1)*n) ] + [(-20, 20) for i in range(2*k-1) ]
    
    # initial guess !
    z0 = np.array(x0 + [0.1 for i in range((2*k-1) * n - 2*n)] + xf + u0 +
                  [1 for i in range(2*k-1 - 2)] + uf)

    # minimizing the objective functional using SLSQP
    opt_res = minimize(
        objective_functional,
        z0,
        method='SLSQP',
        constraints=cons,
        bounds=bnds,
        options={
            'ftol': 0.05,
            'disp': True
        })
    print('================================')
    print('\n \n')
    
    ct.opt_res= opt_res
    # ipydex.IPS()



        
        







