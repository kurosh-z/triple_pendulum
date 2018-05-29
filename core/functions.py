
'''
Core Functions 

Developer:
-----------
Kurosh Zamani

-----------
To Do :
  - solving compatility issues with python 2!
  -
  -


'''
#from __future__ import division, print_function
#=============================================================
# Standard Python modules
#=============================================================
import os, sys

#=============================================================
# External Python modules
#=============================================================

from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)
import sympy as sm
from sympy.solvers import solve
import sympy.physics.mechanics as me
import numpy as np
import mpmath as mp
import scipy as sc 

import config
#=============================================================
# Functions
#=============================================================


def generate_state_equ(mass_matrix, forcing_vector, qdot, qdd, u):
    '''
    given mass_matrix and forcing_vector of a Kane's Method it
    generates state equation :
                         xdot=f(x)+g(x).u
               
                  'ATTENTION : it assumes qdd0 as an Input u'
    
    '''
    xx_dot = sm.Matrix.hstack(sm.Matrix(qdot).T, sm.Matrix(qdd).T).T

    #finding qddot as a function of u (u=qdd0)
    expr = mass_matrix * xx_dot - forcing_vector

    #setting qdd[0] as input u !
    expr = expr.subs(dict([(qdd[0], u)]))

    #finding qddts in respect to qdots and u
    expr_list = [expr[i] for i in range(3, len(qdd) + 2)]
    var_list = [qdd[i] for i in range(1, len(qdd))]

    sol = solve(expr_list, var_list)

    #determining fx and gx
    qdd_expr = [sol[qdd[i]].expand() for i in range(1, len(qdd))]

    #finding terms with and without input term 'u' to determine fx and gx
    collected_qdd = [
        sm.collect(qdd_expr[i], u, evaluate=False)
        for i in range(len(qdd) - 1)
    ]

    fx = sm.zeros(len(xx_dot), 1)
    gx = sm.zeros(len(xx_dot), 1)
    for i in range(len(qdot)):
        fx[i] = qdot[i]
        gx[i] = 0.0
    for i in range(len(qdot) + 1, len(xx_dot)):
        indx = i - (len(qdot) + 1)
        fx[i] = collected_qdd[indx][1]
        gx[i] = collected_qdd[indx][u]
    gx[2] = 1
    return fx, gx


def linearize_state_equ(fx, gx , q, qdot, param_values, equilib_point, numpy_conv=True):
    
    '''
    generate Linearzied form of state Equations in a given equilibrium point :
    
                     xdot= fx + gx.u ---> x_dot= A.delta_x + B.delta_u
     where :
                     A= dfx/dxx|eq.Point + dgx/dxx|eq.Point 
                     B= dg/du|eq.Point
                     
                     dim(A) : (n,n)
                     dim(B) : (n, 1)
     where A and B are linearizations of fx and gx
     
     INPUTS :
     parameter_values : list of tupels --> [(symb1, val1), (symb2, val2), ...]
     equilib_point    : sympy.Matrix   --> sm.Matrix([0.0, 0.0, 1.0, 2.0])
     
     
     
    '''
    #defining values_dict to be substituted in sympy expressions
    qq=sm.Matrix([q, qdot])
    values = list(map(lambda a,b :(a,b),qq, equilib_point)) + param_values
    values_dict=dict(values)
    
    xx=sm.Matrix.hstack(sm.Matrix(q).T, sm.Matrix(qdot).T).T
    fx_lin=fx.jacobian(xx).subs(values_dict)
    gx_lin=gx.jacobian(xx).subs(values_dict)
    
    A=fx_lin+gx_lin
    B=gx.subs(values_dict)
    
    if numpy_conv :
        A= np.array(A.tolist()).astype(np.float64)
        B= np.array(B.tolist()).astype(np.float64)
    
    return A, B


def lqr(A, B, Q, R, additional_outputs=False):
    """
    solve the continous time lqr controller:
    dx/dt = A x + B u
    cost : integral x.T*Q*x + u.T*R*u

    """
    #solving the algebric riccati equation
    P = sc.linalg.solve_continuous_are(A, B, Q, R)
    
    #compute LQR gain
    K = sc.linalg.inv(R) * (B.T.dot(P))
    
    if additional_outputs :
        eigVals, eigVec = sc.linalg.eig(A - B * K)
        ret= K, P, eigVals, eigVec
    else :
        ret =K
        
    
    return ret    


def convert_qdd_to_func(fx, gx, q, qdot, u, param_dict=None):
    
    
    '''
    generate rhs function for pytrajectory
    
    '''
    qdd_expr=fx+gx*u
    
    
    #substituting parameters in qdd_expr 
    if isinstance(param_dict, dict):
        qdd_expr= qdd_expr.subs(param_dict)
    
    #converting qdd_expr to function
    
    #   TO DO :
    #  -resolving compatibility issues with python 2 !!!

    #qdd_func=[sm.lambdify([*q, *qdot, u], qdd_expr[i+len(q)], 'sympy') for i in range(len(q))]
    qdd_func=[sm.lambdify([q[0],q[1] ,qdot[0],qdot[1] , u], qdd_expr[i+len(q)], 'sympy') for i in range(len(q))]
 
    return qdd_func


def pytraj_rhs(x, u, uref=None, t=None, pp=None):
    
    '''
     right hand side function for pytrajecotry  
     YOU SHOULD: run convert_qdd_to_func once before using pytraj_rhs !!!
     
     GOOD TO KNOW :for pytraj rhs function must be defined as a "sympy" function!
     
    '''
    qq = x
    u0, = u
    
    
    #frist defining xd[0 to len(q)] as qdots / ATTENTION: len(qq) = 2*len(q)
    q_dim=int(len(qq)/2)
    xd=[x[i+q_dim] for i in range(q_dim)]
    

    #   TO DO :
    #  -resolving compatibility issues with python 2 !!!

    # xd[len(q)+1 to 2*len(q)] := qddts    
    for i in range(q_dim):
    #    xd.append(qdd_functions[i](*qq,u0))
         xd.append(config.qdd_functions[i](qq[0], qq[1], qq[2], qq[3],u0))


    ret= np.array(xd)

    return ret    

    