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

#=============================================================
# DRICON  :
# =============================================================

ipydex.activate_ips_on_exception()


def collocation_constrains_generator(z):
    '''returns collocation constraints
    '''
    fxu = cfg.pendata.trajectory.fxu
    # n : len(q)
    n = 4
    # number of knotes
    k = 50
    # final time
    h = 2
    hk = h / (k - 1)
    # boundry values for x and u
    #x0 = np.array([0, np.pi, 0, 0])
    #xf = np.array([0, 0, 0, 0])
    #u0 = np.array([0])
    #uf = np.array([0])
    #len_z0 = k * (n + 1) - 2 * (n + 1)
    #indx1 = 0
    #indx2 = len_z0 - (k - 2)
    #z = np.block([x0, z0[indx1:indx2], xf, u0, z0[indx2: ], uf])
    x = z[0:n * k].reshape(k, n)
    u = z[n * k:]
    gz = np.zeros((n * (k - 1), 1))
    for i in range(k - 1):
        x_0 = x[i, :]
        x_1 = x[i + 1, :]
        u_0 = np.array([u[i]])
        u_1 = np.array([u[i + 1]])
        f_0 = fxu(x_0, [u_0])
        f_1 = fxu(x_1, [u_1])
        # computing gradients with this format :
        # df_0= [d/dx_0, d/dx_1, d/du0, d/du1]
        # df_x0u0= dfxu(x_0, u_0)
        # df_0= np.block([df_x0u0[:,0:n],np.zeros((n,n)), df_x0u0[:,-1, np.newaxis] ,  np.zeros((n,1))])
        # df_x1u1= dfxu(x_1, u_1)
        # df_1= np.block([np.zeros((n,n)), df_x1u1[:,0:n], np.zeros((n,1)), df_x0u0[:,-1, np.newaxis]])
        # collocation point and gradient in the same matrix format
        x_c = 0.5 * (x_0 + x_1) + (hk / 8.0) * (f_0 - f_1)
        # dx_c = np.block([.5*np.eye(n) , .5*np.eye(n) , np.zeros((n, 2))]) + h/8*(df_0 - df_1)
        # control u_c and gradient in matrix format
        u_c = 0.5 * (u_0 + u_1)
        # du_c= np.array([0 for j in range(2*n)] + [0.5 , 0.5])
        #  xc of the spline and its gradient
        xdot_c = -1.5 / h * (x_0 - x_1) - 1 / 4.0 * (f_0 + f_1)
        # dxdot_c= -3/2/h* np.block([np.eye(n), -1 * np.eye(n), np.zeros((n, 2))]) - 1/4*(df_0 + df_1)
        # calcualte pendulum dynamics at the collocation point and gradient
        f_c = fxu(x_c, u_c)
        # df_c= dfxu(x_c, u_c)
        # add to the gz  vector
        gz[i * n:(i + 1) * n, :] = (xdot_c - f_c)[::, np.newaxis]
    return np.ravel(gz)


def objective_func(z):
    '''returns objective functional to be minimized
    '''

    fxu = cfg.pendata.trajectory.fxu
    k = 50
    n = 4
    h = 2
    hk = h / (k - 1)

    #x0 = np.array([0, np.pi, 0, 0])
    xf = z[len(z) - (n + k):len(z) - k]
    #u0 = np.array([0])
    #uf = np.array([0])
    #len_z0 = k * (n + 1) - 2 * (n + 1)
    # indx1 = 0
    #indx2 = len_z0 - (k - 2)
    #z = np.block([x0, z0[indx1:indx2], xf, u0, z0[indx2 :], uf])
    S = Q = np.eye(4)
    R = 0.01 * np.identity(1)
    Ju = 0.5 * xf.dot(S).dot(xf)
    lxut = lambda x, u: 0.5 * x.dot(Q).dot(x) + 0.5 * u.dot(R).dot(u)
    x = z[0:n * k].reshape(k, n)
    u = z[n * k:]
    # iterate over x and u and calculate the objective Ju
    for i in range(k - 1):
        # constructing x_k and x_k+1 and integrate l(x,u,t) over
        #  the time period between thease two points using sympson quadrature!
        x_0 = x[i, :]
        x_1 = x[i + 1, :]
        u_0 = np.array([u[i]])
        u_1 = np.array([u[i + 1]])
        f_0 = fxu(x_0, u_0)
        f_1 = fxu(x_1, u_1)
        x_c = 0.5 * (x_0 + x_1) + hk / 8.0 * (f_0 - f_1)
        u_c = 0.5 * (u_0 + u_1)
        Ju += hk / 6.0 * (lxut(x_0, u_0) + 4 * lxut(x_c, u_c) + lxut(x_1, u_1))
    return Ju


def trajectory_generator(ct, max_time):
    ''' trajectory optimization using DRICON algorithm

    article : https://ieeexplore.ieee.org/document/7487270/
    '''

    print('\n \n')
    print(
        '======================== Trajectory_generator ========================'
    )
    q = ct.model.q
    k = 50
    n = 4
    h = max_time

    x0 = [0.0] + [np.pi
                  for i in range(len(q) - 1)] + [0.0 for i in range(len(q))]
    xf = [0.0 for i in range(2 * len(q))]

    # ipydex.IPS()

    u0 = [0.0]
    uf = [0.0]

    # fxu needet
    fxu = sympy_states_to_func()
    ct.trajectory.fxu = fxu
    # dfxu = jac_func(ct)

    # z= [q00 , qdot00, q10, qdot10, ... q1n, qdot1n, u1, u2, ... un, lambda1, ... lambda_n-1 ,gama1, ... , gama_n-1]

    # defining constraints
    collocation_cons = {'type': 'eq', 'fun': collocation_constrains_generator}

    # boundry conditions
    boundry_x0 = {'type': 'eq', 'fun': lambda z: z[:4] - np.array(x0)}
    boundry_u0 = {'type': 'eq', 'fun': lambda z: z[k * n] - np.array(u0)}
    boundry_xf = {'type': 'eq', 'fun': lambda z: z[k*n - n:k*n] - np.array(xf)}
    boundry_uf = {'type': 'eq', 'fun': lambda z: z[-1] - np.array(uf)}

    # all constrainsts together
    # cons = (collocation_cons, boundry_x0, boundry_u0, boundry_xf, boundry_uf)
    cons = (collocation_cons, boundry_x0, boundry_xf)

    # initial guess !
    z0 = np.array(x0 + [0.1 for i in range(k * n - 8)] + xf + u0 +
                  [1 for i in range(k - 2)] + uf)

    # minimizing the objective functional using SLSQP
    opt_res = minimize(
        objective_func,
        z0,
        method='SLSQP',
        constraints=cons,
        options={
            'ftol': 0.05,
            'disp': True
        })
    print('================================')
    print('\n \n')

    # assembling the splines together and make x_s an   qwqd u_s
    def x_s(t):
        ''' makes the x_traj function from splines
            and returns x_traj for time t !
        '''
        opt_res = cfg.pendata.trajectory.opt_res
        fxu = cfg.pendata.trajectory.fxu
        k = 50
        n = 4
        
        z = opt_res.x
        x = z[0:n * k].reshape(k, n)
        u = z[n * k:]

        # z = opt_res.x.reshape((k, n + 1))
        # x = z[:, :-1]
        # u = z[:, -1]

        hk = float(max_time) / (k - 1)
        indx = int(t / hk)

        if indx >= k - 2:
            indx = k - 2

        x_0 = x[indx, :]
        x_1 = x[indx + 1, :]
        u_0 = [u[indx]]
        u_1 = [u[indx + 1]]
        f_0 = fxu(x_0, u_0)
        f_1 = fxu(x_1, u_1)

        xs = lambda t: x_0 + f_0 * t + (3 / hk**2 * (x_1 - x_0) - 1 / hk * (2 * f_0 + f_1)) * t**2 + (2 / hk**3 * (x_0 - x_1) + 1 / hk**2 * (f_0 + f_1)) * t**3

        tk = indx * hk
        tau = t - tk

        x_traj = xs(tau)

        return x_traj

    def u_s(t):
        ''' all segment splines of u  together as 
            a function of time
        '''

        opt_res = cfg.pendata.trajectory.opt_res


        z= opt_res.x
        # x = z[0:n * k].reshape(k, n)
        u = z[n * k:]
        # z = opt_res.x.reshape((k, n + 1))
        # u = z[:, -1]
        
        hk = float(max_time) / (k - 1)        
        indx = int(t / hk)
        if indx > k-2 :
            indx= k-2 

        tk = indx * hk
        tau = t - tk

        u_0 = u[indx]
        u_1 = u[indx + 1]
        uc = 0.5 * (u_0 + u_1)
        us= 2.0 / hk**2 * (tau- hk/2) * (tau- hk) * u_0 - 4.0 / hk**2 * (tau)*(tau-hk)*uc + 2.0 / hk**2 * tau * (tau-hk/2.0)*u_1


        return np.array([us])

    ct.trajectory.opt_res = opt_res
    ct.trajectory.cs_ret = (x_s, u_s)
    ct.trajectory.max_time = max_time
    ct.trajectory.xa = x0
    ct.trajectory.xb = xf
    ct.trajectory.ua = u0
    ct.trajectory.ub = uf

    ipydex.IPS()
