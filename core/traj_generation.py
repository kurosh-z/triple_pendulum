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

    x = z[0:(2 * k - 1) * n].reshape(2 * k - 1, n)
    x_knot = x[::2]
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

    x = z[0:(2 * k - 1) * n].reshape(2 * k - 1, n)
    x_knot = x[::2]
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
    fxu=cfg.pendata.trajectory.fxu
    max_time = cfg.pendata.trajectory.max_time
    hk = max_time / (k - 1)

    x = z[0:(2 * k - 1) * n].reshape(2 * k - 1, n)
    x_knot = x[::2]
    xc = x[1::2]

    # u_knote: uk on each knote , uc: u(k+1/2)
    u_knot = z[(2 * k - 1) * n::2]
    uc = z[(2 * k - 1) * n + 1::2]

    xf = x[-1]

    S = Q = np.eye(4)
    R = 1.0 * np.identity(1)
    Ju = 0.5 * xf.dot(S).dot(xf)

    S2=S



    # defining l(x,u, t)
    lxut = lambda x, u: 0.5 * x.dot(Q).dot(x) + 0.5 * u.dot(R).dot(u)
    epsilon =0.0
    for i in range(k - 1):

        x0 = x_knot[i]
        x1 = x_knot[i + 1]
        x01 = xc[i]

        u0 = np.array([u_knot[i]])
        u1 = np.array([u_knot[i + 1]])
        u01 = np.array([uc[i]])

        f0 = fxu(x0, u0)
        f1 = fxu(x1, u1)
        f01 = fxu(x01, u01)

        xspl = x0 + hk / 6.0 * (f0 + 4 * f01 + f1)

        epsilon += (xspl - x01).dot(S2).dot(xspl- x01)
        print('eps :',epsilon)

        # Updating our J(u)
        Ju += hk / 6.0 * (lxut(x0, u0) + 4 * lxut(x01, u01) + lxut(x1, u1)) 
        print('Ju:', Ju)
        

    return Ju

def error_function(z):
    k = cfg.pendata.trajectory.k
    n = cfg.pendata.trajectory.n
    fxu=cfg.pendata.trajectory.fxu
    max_time = cfg.pendata.trajectory.max_time
    hk = max_time / (k - 1)
    x = z[0:(2 * k - 1) * n].reshape(2 * k - 1, n)
    x_knot = x[::2]
    xc = x[1::2]

    # u_knote: uk on each knote , uc: u(k+1/2)
    u_knot = z[(2 * k - 1) * n::2]
    uc = z[(2 * k - 1) * n + 1::2]

    S2= np.eye(4)
    epsilon=0.0
    for i in range(k - 1):
    
        x0 = x_knot[i]
        x1 = x_knot[i + 1]
        x01 = xc[i]

        u0 = np.array([u_knot[i]])
        u1 = np.array([u_knot[i + 1]])
        u01 = np.array([uc[i]])

        f0 = fxu(x0, u0)
        f1 = fxu(x1, u1)
        f01 = fxu(x01, u01)

        xspl = x0 + hk / 6.0 * (f0 + 4 * f01 + f1)

        epsilon += (xspl - x01).dot(S2).dot(xspl- x01)
    return epsilon    

        




def trajectory_generator(ct, max_time):
    ''' generates trajectory using collocation method
    '''
    print('\n \n')
    print(
        '======================== Trajectory_generator ========================'
    )

    q = ct.model.q
    ct.trajectory.n = n = 2 * len(q)
    ct.trajectory.k = k = 20

    ct.trajectory.max_time = h = max_time

    x0 = [0.0] + [np.pi
                  for i in range(len(q) - 1)] + [0.0 for i in range(len(q))]
    xf = [0.0 for i in range(2 * len(q))]
    # xf= [0.01, 10*np.pi/3, 0, 0 ]

    u0 = [0.0]
    uf = [0.0]

    # fxu needet for constraint generator
    fxu = sympy_states_to_func()
    ct.trajectory.fxu = fxu

    # defining constraints
    collocation_cons = {'type': 'eq', 'fun': collocation_constrains}
    interpolation_cons = {'type': 'eq', 'fun': interpolation_constrains}
    error_cons= {'type': 'ineq', 'fun': lambda z : error_function(z) -10**-6 }

    # boundry conditions
    boundry_x0 = {'type': 'eq', 'fun': lambda z: z[:n] - np.array(x0)}
    boundry_u0 = {
        'type': 'eq',
        'fun': lambda z: z[(2 * k - 1) * n] - np.array(u0)
    }
    boundry_xf = {
        'type': 'eq',
        'fun': lambda z: z[(2 * k - 1) * n - n:(2 * k - 1) * n] - np.array(xf)
    }
    boundry_uf = {'type': 'eq', 'fun': lambda z: z[-1] - np.array(uf)}

    # def x_boundry_generator(i,j,x):
    #     ''' returns a function for x[i] boundry condition
    #         you cann loop over it and make scalar boundry
    #         conditions for x[i] i = from 0 to n

    #         that's just a trick to turn a vector boundry condition
    #         to scalar boudnry conditions of each elements !
    #     '''
    #     x_boundry_func = lambda z, i=i, j=j : z[i]-x[j]

    #     return x_boundry_func

    # boundry_x0_tupel= ()
    # for i in range (n) :

    #     boundry_x0_dict = {'type': 'eq', 'fun': x_boundry_generator(i,i,x0)}
    #     boundry_x0_tupel+= (boundry_x0_dict, )

    # boundry_xf_tupel= ()
    # xf_in_z= np.arange((2*k-1)*n - n, (2*k-1)*n)
    # xf_in_xf=np.arange(0, n)
    # for i,j in  zip(xf_in_z, xf_in_xf) :

    #     boundry_xf_dict = {'type': 'eq', 'fun': x_boundry_generator(i,j,xf)}
    #     boundry_xf_tupel+= (boundry_xf_dict, )

    # all constrainsts together
    cons = (collocation_cons, interpolation_cons, boundry_x0, boundry_xf, error_cons)
    #cons = (collocation_cons, interpolation_cons,boundry_x0, boundry_xf)

    # cons= (collocation_cons,interpolation_cons) + boundry_x0_tupel + boundry_xf_tupel

    bnds = [(None, None)
            for i in range((2 * k - 1) * n)] + [(-20, 20)
                                                for i in range(2 * k - 1)]

    # initial guess !
    z0 = np.array(x0 + [0.1 for i in range((2 * k - 1) * n - 2 * n)] + xf +
                  u0 + [1 for i in range(2 * k - 1 - 2)] + uf)

    # minimizing the objective functional using SLSQP
    opt_res = minimize(
        objective_functional,
        z0,
        method='SLSQP',
        constraints=cons,
        options={
            'ftol': 0.001,
            'disp': True
        })
    print('================================')
    print('\n \n')

    ct.trajectory.opt_res = opt_res

    def x_s(t):
        ''' makes the x_traj function from splines
            and returns x_traj for time t !
        '''
        ct = cfg.pendata
        opt_res = ct.trajectory.opt_res
        fxu = ct.trajectory.fxu
        n = ct.trajectory.n
        k = ct.trajectory.k
        max_time = ct.trajectory.max_time

        x = opt_res.x[:(2 * k - 1) * n].reshape(2 * k - 1, n)
        xk = x[::2]
        xc = x[1::2]
        u = opt_res.x[(2 * k - 1) * n:]
        uk = u[::2]
        uc = u[1::2]

        hk = float(max_time) / (k - 1)
        

        indx = int(t / hk)
        # if t == hk :
        #     indx+=-1

        # if t == 2*hk :
        #     indx+=-1

        if indx >= k - 2:
            indx = k - 2

        x_0 = xk[indx]
        x_01 = xc[indx]
        x_1 = xk[indx + 1]

        u_0 = [uk[indx]]
        u_01 = [uc[indx]]
        u_1 = [uk[indx + 1]]

        f_0 = fxu(x_0, u_0)
        f_01 = fxu(x_01, u_01)
        f_1 = fxu(x_1, u_1)

        tk = indx * hk
        tau = t - tk

        xs = x_0 + f_0 * (tau / hk) + 0.5 * (-3.0 * f_0 + 4. * f_01 - f_1) * (
            tau / hk)**2 + 1.0 / 3.0 * (2 * f_0 - 4 * f_01 + 2 * f_1) * (
                tau / hk)**3

        ta = 0.0
        tc = hk 
        tb = (ta + tc) / 2.0
        xdot_func = lambda tt: ((tt - tb) * (tt - tc) / ((ta - tb) * (ta - tc))) * f_0 + ((tt - ta) * (tt - tc) / ((tb - ta) * (tb - tc))) * f_01 + ((tt - ta) * (tt - tb) / ((tc - ta) * (tc - tb))) * f_1

        xspl = x_0 + tau / 6.0 * (xdot_func(0.0) + 4 * xdot_func(tau/2) + xdot_func(tau))

        import scipy.integrate as integrate
        q1dot_func = lambda tt:(((tt - tb) * (tt - tc) / ((ta - tb) * (ta - tc))) * f_0 + ((tt - ta) * (tt - tc) / ((tb - ta) * (tb - tc))) * f_01 + ((tt - ta) * (tt - tb) / ((tc - ta) * (tc - tb))) * f_1)[1] 
        xspl2= x_0[1] + integrate.quad(q1dot_func, 0, tau)[0]

        print('=====debuging ======')
        print('t     : ', t)
        print('tau   : ',tau)
        print('indx  : ', indx)
        print('ta    : ', ta)
        print('tb    : ', tb)
        print('tc    : ', tc)
        print('f_0   : ', f_0 )
        print('xdot00: ',xdot_func(0.0) )
        print('f_01  : ', f_01)        
        print('xdot01: ',xdot_func(tb) )
        print('f_1   : ', f_1 )
        print('xdot1 : ' , xdot_func(tc))
        print('x_0   : ', x_0)
        print('x_01  : ', x_01)
        print('x_1   : ', x_1)
        print('xspl  : ', xspl)
        print('xspl2 : ', xspl2)

        print('======================')
        print('\n \n')

        return xspl


    def u_s(t):
        ''' makes the x_traj function from splines
            and returns x_traj for time t !
        '''
        ct = cfg.pendata
        opt_res = ct.trajectory.opt_res
        fxu = ct.trajectory.fxu
        n = ct.trajectory.n
        k = ct.trajectory.k
        max_time = ct.trajectory.max_time

        u = opt_res.x[(2 * k - 1) * n:]
        uk = u[::2]
        uc = u[1::2]

        hk = float(max_time) / (k - 1)

        indx = int(t / hk)
        


        if indx >= k - 2:
            indx = k - 2


        u_0 = uk[indx]
        u_01 = uc[indx]
        u_1 = uk[indx + 1]

        tk = indx * hk
        tau = t - tk


        us= 2.0 / hk**2 * (tau- hk/2) * (tau- hk) * u_0 - 4.0 / hk**2 * (tau)*(tau-hk)*u_01 + 2.0 / hk**2 * tau * (tau-hk/2.0)*u_1
        
        ta = hk * indx
        tc = hk * (indx + 1)
        tb = (ta + tc) / 2.0
        u_func = lambda tt: ((tt - tb) * (tt - tc) / ((ta - tb) * (ta - tc))) * u_0 + ((tt - ta) * (tt - tc) / ((tb - ta) * (tb - tc))) * u_01 + ((tt - ta) * (tt - tb) / ((tc - ta) * (tc - tb))) * u_1

        uspl = u_0 + hk / 6.0 * (u_func(0.0) + 4 * u_func(tau/2) + u_func(tau))

        return np.array([uspl])




    cs_ret = ct.trajectory.cs_ret= (x_s, u_s)
    ct.trajectory.xa = x0
    ct.trajectory.xb = xf
    ct.trajectory.ua = u0
    ct.trajectory.ub = uf

    
    tvec = np.linspace(0, max_time, 2 * k - 1)
    tvec2= np.linspace(0, max_time, 6*k-1)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    theta = opt_res.x[:(2 * k - 1) * n].reshape(2 * k - 1, n)[:, 1]
    u = opt_res.x[(2 * k - 1) * n:]
    y = opt_res.x[:(2 * k - 1) * n].reshape(2 * k - 1, n)[:, 0]

    # theta_spline= np.array([x_s(t)[1] for t in tvec2])
    # u_splines= np.array([u_s(t)][0] for t in tvec2)

    plt.subplot(2, 1, 1)
    plt.plot(tvec, theta * 180 / np.pi, 'o')
    plt.ylabel('q1')

    # plt.subplot(4,1,2)
    # plt.plot(tvec2, theta_spline, '-.')


    plt.subplot(2, 1, 2)
    plt.plot(tvec, u, 'o')
    plt.ylabel('u')

    # plt.subplot(4,1,4)
    # plt.plot(tvec2, u_splines, '-.')

    plt.show()

    # ipydex.IPS()

