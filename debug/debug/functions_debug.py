'''
Core Functions 

Developer:
-----------
Kurosh Zamani

-----------
To Do :
  - solving compatility issues with python 2!
    ( Python2 :function pytraj_rhs is still not generally
     working with n other than 1! )
  - its better to merge state_functions and convert_qdd_to_func
    into a single function !


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

from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)
import sympy as sm
from sympy.solvers import solve
import sympy.physics.mechanics as me
import numpy as np
from numpy.linalg import inv as np_inv
from scipy import linalg

import config

import ipydex
# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

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


def linearize_state_equ(fx,
                        gx,
                        q,
                        qdot,
                        u,
                        param_values,
                        equilib_point=None,
                        numpy_conv=True):
    '''
    generate Linearzied form of state Equations at a given 
    equilibrium point.
    if equilib_point is None it returns A and B as sympy expressions
    of x0 and u0 !
    
         xdot= fx + gx.u ---> x_dot= A.delta_x + B.delta_u
     where :
              A= dfx/dxx|eq.Point + dgx/dxx|eq.Point 
              B= dg/du|eq.Point
                     
                     dim(A) : (n,n)
                     dim(B) : (n, 1)
    
    =========================================================
     INPUTS :
    =========================================================
     parameter_values : list of tupels --> [(symb1, val1), (symb2, val2), ...]
     equilib_point    : a list containing --> [x0 , u0]  
                    

    ===========================================================
     OUTPUTS:
    =========================================================== 
     By default "numpy.array" A , B
     if nump_conv is False or no equilibrium point is given 
     it returns "sympy.Matrix"  A , B
     
    '''
    #defining values_dict to be substituted in sympy expressions
    if equilib_point is None:
        values_dict = dict(param_values)
        numpy_conv = False
    else:
        qq = q + qdot + [u]
        values = zip(qq, equilib_point) + param_values
        values_dict = dict(values)

    xx = sm.Matrix.vstack(sm.Matrix(q), sm.Matrix(qdot))
    df_dxx = fx.jacobian(xx)
    dg_dxx = gx.jacobian(xx)

    A = (df_dxx + dg_dxx * u).subs(values_dict)
    B = gx.subs(values_dict)

    if numpy_conv:
        A = np.array(A.tolist()).astype(np.float64)
        B = np.array(B.tolist()).astype(np.float64)

    return A, B


def lqr(A, B, Q, R, additional_outputs=False):
    """
    solve the continous time lqr controller:
    dx/dt = A x + B u
    cost : integral x.T*Q*x + u.T*R*u

    """
    #solving the algebric riccati equation
    P = linalg.solve_continuous_are(A, B, Q, R)
    #compute LQR gain
    k = np_inv(R) * (B.T).dot(P)

    if additional_outputs:
        eigVals, eigVec = linalg.eig(A - B * k)
        ret = k, P, eigVals, eigVec
    else:
        ret = k

    return ret

"""
old_working : 
def convert_qdd_to_func(fx, gx, q, qdot, u, param_dict=None):
    '''
    convert state equations form sympy to function
    output is a list containing [qdd0, qdd1, qdd2, ... ]

    TO DO:
     - change it to be able to work with every n in python2
    '''
    qdd_expr = fx + gx * u

    #substituting parameters in qdd_expr
    if isinstance(param_dict, dict):
        qdd_expr = qdd_expr.subs(param_dict)

    #converting qdd_expr to function

    #   TO DO :
    #  -resolving compatibility issues with python 2 !!!

    #qdd_func=[sm.lambdify([*q, *qdot, u], qdd_expr[i+len(q)], 'sympy') for i in range(len(q))]
    qdd_func = [
        sm.lambdify([q[0], q[1], qdot[0], qdot[1], u], qdd_expr[i + len(q)],
                    'sympy') for i in range(len(q))
    ]

    return qdd_func
"""
def convert_qdd_to_func(fx, gx, q, qdot, u, param_dict=None):
    '''
    convert state equations form sympy to functions
    output is a list containing [qdd0, qdd1, qdd2, ... ]

    '''
    qdd_expr = fx + gx * u

    #substituting parameters in qdd_expr
    if isinstance(param_dict, dict):
        qdd_expr = qdd_expr.subs(param_dict)

    #converting qdd_expr to function
    qdd_func = [
        sm.lambdify(q + qdot + [u], qdd_expr[i + len(q)], 'sympy')
        for i in range(len(q))
    ]

    return qdd_func
"""
old working :
def pytraj_rhs(x, u, uref=None, t=None, pp=None):
    '''
     right hand side function for pytrajecotry  
     YOU SHOULD: run convert_qdd_to_func once before using pytraj_rhs !!!
     
     GOOD TO KNOW :for pytraj rhs function must be defined as a "sympy" function!
     
    '''
    qq = x
    u0, = u

    #frist defining xd[0 to len(q)] as qdots / ATTENTION: len(qq) = 2*len(q)
    q_dim = int(len(qq) / 2)
    xd = [x[i + q_dim] for i in range(q_dim)]

    #   TO DO :
    #  -resolving compatibility issues with python 2 !!!

    # xd[len(q)+1 to 2*len(q)] := qddts
    for i in range(q_dim):
        #    xd.append(qdd_functions[i](*qq,u0))
        xd.append(config.qdd_functions[i](qq[0], qq[1], qq[2], qq[3], u0))

    ret = np.array(xd)

    return ret
"""
def pytraj_rhs(x, u, uref=None, t=None, pp=None):
    '''
     right hand side function for pytrajecotry  
     YOU SHOULD: run convert_qdd_to_func once before using pytraj_rhs !!!
     
     GOOD TO KNOW :for pytrajectory rhs function must be defined as a 
                   "sympy" function and not "numpy" !
     
    '''
    qq = x
    u0, = u

    # first defining xd[0 to len(q)] as qdots / ATTENTION: len(qq) = 2*len(q)
    q_len = int(len(qq) / 2)
    xd = [x[i + q_len] for i in range(q_len)]

    #   TO DO :
    #  -resolving compatibility issues with python 2 !!!

    # xd[len(q)+1 to 2*len(q)] := qddts
    for i in range(q_len):
        #    xd.append(qdd_functions[i](*qq,u0))
        if len(qq) == 4:
            xd.append(config.qdd_functions[i](qq[0], qq[1], qq[2], qq[3], u0))
        elif len(qq) == 6:
            xd.append(config.qdd_functions[i](qq[0], qq[1], qq[2], qq[3],
                                              qq[4], qq[5], u0))
        elif len(qq) == 8:
            xd.append(config.qdd_functions[i](qq[0], qq[1], qq[2], qq[3],
                                              qq[4], qq[5], qq[6], qq[7], u0))

    ret = np.array(xd)

    return ret

# riccuti differential equation
def riccati_diff_equ(P, t, A_equi, B_equi, Q, R, dynamic_symbs):
    '''
    =========================================================
    DESCRIPTION :
    it returns riccati differential equation :

      P_dot = -P * A - A.T * P + P * b * R**-1 * b.T * P - Q
    
    =========================================================

     INPUTS :
    
    - P
    - t
    - A_equi  : sympy.Matrix 
    - B_equi  : sympy.Matrix 
    - Q
    - R
    - dynamic_symbs : list containing symbolic variable in A_equi 
                      and B_equi .
        example --->  dynamic_symb= [q0, q1, qdot0, qdot1, u]
                
                
    ===========================================================

     OUTPUTS:
     
    P_dot : numpy.array 
     
    '''
    # evaluating A_equi and B_equi at equilibrium point
    logging.debug('P_new: %s \n \n', P)

    logging.debug('Debuging Message from riccati_diff_equ')
    logging.debug('---------------------------------------------------')
    
    logging.debug('t : %f \n', t)
    if t<0 :
        x0=[0, np.pi, 0, 0]
        u0=[0]
    else:    
        x0 = config.cs_ret[0](t)
        u0 = config.cs_ret[1](t)
    
    logging.debug('x0 : %s', x0)
    logging.debug('u0 : %s \n', u0)
    

    equilib_point = np.hstack((x0, u0))
    values_dict = dict(zip(dynamic_symbs, equilib_point))

    A = A_equi.subs(values_dict)
    B = B_equi.subs(values_dict)

    # converting sympy.Matrix to numpy.array
    A = np.array(A.tolist()).astype(np.float64)
    B = np.array(B.tolist()).astype(np.float64)
    
    logging.debug('P_current: %s', P)

    P = P.reshape((4, 4))
    P_dot = -P.dot(A) - (A.T).dot(P) + P.dot(B.dot(np_inv(R) *
                                                   (B.T).dot(P))) - Q

    ret = P_dot.reshape(16)
    logging.debug('p_dot: %s', ret)    
    return ret


# lqr_tracking function :
def generate_gain_matrix(R, B_equi, P, Vect, dynamic_symbs):
    '''
    returns input 'u' for tracking 
    each row of K_matrix include gain k at time t
     -->  num_rows= len(Vect), num_columns= 4
                 
    TO DO :
     -change it so that it ables to function with every n !
    '''
    K_matrix = np.zeros((len(Vect), 4))

    for i, t in enumerate(Vect):
        P_eq = P[i, :].reshape(4, 4)

        # evaluating  B_equi at equilibrium point
        x0 = config.cs_ret[0](t)
        u0 = config.cs_ret[1](t)
        equilib_point = np.hstack((x0, u0))
        values_dict = dict(zip(dynamic_symbs, equilib_point))

        B = B_equi.subs(values_dict)
        # converting B (sympy.Matrix) to numpy.array
        B = np.array(B.tolist()).astype(np.float64)

        # finding k and add it to K_matrix
        K_matrix[i, :] = np_inv(R) * B.T.dot(P_eq)

    return K_matrix


# converting symbolic state equations to functions
def sympy_states_to_func(dynamic_symbs, param_list):
    '''
    it converts symbolic state equations to functions

    ATTENTION :
       - the difference between this and qdd_to_func is
         that qdd_to_func returns 'sympy' funcitons, 
         sympy_states_to_func on the other hand returns
         numpy functions !
       -it uses fx, gx and parameter_values stored in config.py
        fx and gx are calculated in sys_model.py and stored in 
        config.fx_expr and config.gx_expr
    ============================================================

    INPUTS:
    - dynamic_symbs : an iterable object containing symbolic 
      variables : ---> dynami_symbs= [q0, q1, qdot0, qdot1, u] 
    - param_list :  

    OUTPUTS :
    - callable state_func(state, input) 
    '''
    # making a dictionary of parameters with dynamic_symbs as keys
    param_dict = dict(param_list)

    u = dynamic_symbs[-1]
    num_states = len(dynamic_symbs) - 1

    # logging.debug('u : %s', u)
    
    fx = config.fx_expr
    gx = config.gx_expr
    # logging.debug('fx_expr: %s', fx)
    # logging.debug('gx_expr: %s', gx)

    xdot_expr = (fx + gx * u).subs(param_dict)
    # logging.debug('xdot_exr: %s', xdot_expr)
    
    xdot_func = [
        sm.lambdify(dynamic_symbs, xdot_expr[i], modules='numpy')
        for i in range(num_states)
    ]
    
    def state_func(state, inputs):
        '''
        given state and inputs it return x_dot
        '''
        u, = inputs
        x = state
        xd0=  xdot_func[0](x[0], x[1], x[2], x[3], u)
        xd1=  xdot_func[1](x[0], x[1], x[2], x[3], u)
        xd2=  xdot_func[2](x[0], x[1], x[2], x[3], u)
        xd3=  xdot_func[3](x[0], x[1], x[2], x[3], u)

        x_dot = np.array([xd0, xd1, xd2, xd3])
        
        return x_dot

    return state_func


# tracking function :
# def tracking(K_matrix, Vect):
#     '''
#     using K_matrix it calculates the system_states x :
#           xdot= fx + gx * (-K.T*x)

#     ATTENTION : it uses qdd_fucntions stored in config
#                 (it stored there if you already run traj_opt.py !)
#     '''