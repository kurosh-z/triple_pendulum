# coding: utf-8
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
#from __future__ import deviation, print_function
#=============================================================
# Standard Python modules
#=============================================================
import os, sys, glob
import logging
from multiprocessing import Pool
import multiprocessing

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

# from pytrajectory import aux

import dill
import ipydex

import cfg

# logging.basicConfig(level=logging.DEBUG)

# logger = logging.getLogger()
# handler = logging.StreamHandler()
# formatter = logging.Formatter(
#     '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)

#=============================================================
# Functions
#=============================================================


def jac_func(ct):
    ''' returns jacobinan function of fxu for DRICON
    '''
    q = ct.model.q
    qdot = ct.model.qdot
    u = ct.model.u
    fx = ct.model.fx
    gx = ct.model.gx

    xx = sm.Matrix([q + qdot + [u]])
    # we just need qdds so:
    fxu = sm.Matrix((fx + gx * u))
    jac = fxu.jacobian(xx)

    # to avoid lambdify's bug we need to lambdify every single component
    jac_funcs = [
        sm.lambdify(xx, jac[i, j], 'numpy') for i in range(2 * len(q))
        for j in range(2 * len(q) + 1)
    ]

    def jac_func(x, u):
        ''' returns numpy array of jacobian evaluated at x and u
            Inputs :x and u should be given as numpy array
        '''
        n = int(len(x))
        xx = np.array(x.tolist() + u.tolist())
        jac = np.array(
            [jac_funcs[i](*xx) for i in range(n * (n + 1))]).reshape(n, n + 1)

        return jac

    return jac_func


def generate_state_equ(mass_matrix44, forcing_vector44, dynamic_symbs):
    '''
    given mass_matrix and forcing_vector of a Kane's Method it
    generates state equation :
                         xdot=f(x)+g(x).u
               
                  'ATTENTION : it assumes qdd0 as an Input(u)'
    

    '''
    u = dynamic_symbs[-1]
    len_q = int((len(dynamic_symbs) - 1) / 2)
    qdot= dynamic_symbs[len_q:-1]

    mass_matrix33= mass_matrix44[1:,1:]
    forcing_vector33= forcing_vector44[1:,0]

    mass_matrix33_det = mass_matrix33.det()
    Det = sm.Symbol("Det")
    mass_matrix33_inv = mass_matrix33.adjugate()/Det
    phi_dd = mass_matrix33_inv * (-u * mass_matrix44[1:, 0] + forcing_vector33)
    phi_dd_final= phi_dd.subs(Det, mass_matrix33_det)


    u_terms= phi_dd_final.jacobian(sm.Matrix([u]))
    other_terms= phi_dd_final - u_terms*u
    # ATTENTION: if you look at the other_terms(terms without u!)
    #            there are still terms with u, with coeffcients of order 10e-6
    #            to make sure that it has no negative effects in numerical calculations
    #            we set them manually to zero ! :
    other_terms= other_terms.subs(u, 0.0)


    # defining fx and gx :
    fx = sm.zeros(2 * len_q, 1)
    gx = sm.zeros(2 * len_q, 1)

    for i in range(len_q):
        fx[i] = qdot[i]

    for i in range(len_q + 1, 2 * len_q):
        indx = i - (len_q + 1)
        fx[i] = other_terms[indx]
        gx[i] = u_terms[indx]

    gx[len_q] = 1


    return fx, gx





def generate_state_equ_alt(mass_matrix, forcing_vector, qdot, qdd, u):
    '''
    given mass_matrix and forcing_vector of a Kane's Method it
    generates state equation :
                         xdot=f(x)+g(x).u
               
                  'ATTENTION : it assumes qdd0 as an Input u'
    

    '''
    len_q = len(qdd)
    expr = mass_matrix * sm.Matrix(qdot + qdd).subs(dict([(qdd[0], u)]))

    # expr_u = expr[len_q] - forcing_vector[len_q]
    expr_qdd = [(expr[i] - forcing_vector[i]).expand()
                for i in range(len_q + 1, 2 * len_q)]
    # using collect to find the terms with qdd for each raw
    collected_qdd_expr = [
        sm.collect(expr_qdd[i], qdd[j], evaluate=False)[qdd[j]]
        for i in range(len_q - 1) for j in range(1, len(qdd))
    ]

    # storing coefficents of qdds to a matrix

    # --> for example finding a1, a2 in expr[0] :expr[0]= a1*qdd[1] + a2*qdd[2]
    # each raw of qdd_coeff_matrix include coeff. for each raw of qdd
    qdd_coeff_matrix = sm.ImmutableMatrix(collected_qdd_expr).reshape(
        len(qdd) - 1,
        len(qdd) - 1)

    # simplifying the matrix !
    qdd_coeff_matrix = sm.trigsimp(qdd_coeff_matrix)

    # qdd_vect is just qdd without qdd[0] !
    qdd_vect = sm.ImmutableMatrix([qdd[i] for i in range(1, len(qdd))])

    # finding terms without qdd thease are the constant vector
    qdd_const_vector = sm.ImmutableMatrix(
        expr_qdd) - qdd_coeff_matrix * qdd_vect

    print('starting simplification')
    # simplifying the results ! we want the constants on the
    # other side of the equations so we should multiply it by -1 !
    qdd_const_vector = (-1) * sm.trigsimp(qdd_const_vector)
    # qdd_const_vector = (-1) * qdd_const_vector
    print('we are calculating qdd, its gonna take a while !')

    # solving the linear system for qddots :
    # --->  qdd_coeff_matrix * qdd_vect = qdd_const_vector1 + qdd_const_vector2
    #                         qdd_const_vector1 : inculde terms without u
    #                         qdd_const_vector2 : inlcude terms with u

    collected_qdd_const_vector = [
        sm.collect(qdd_const_vector[i].expand(), u, evaluate=False)
        for i in range(len_q - 1)
    ]

    qdd_const_vector1 = sm.ImmutableMatrix(
        [collected_qdd_const_vector[i][1] for i in range(len_q - 1)])
    qdd_const_vector2 = sm.ImmutableMatrix(
        [collected_qdd_const_vector[i][u] for i in range(len_q - 1)])
    print('inversing qdd_coeff_matrix')
    qdd_coeff_matrix_inv = qdd_coeff_matrix.inv()
    print('inversion of qdd_coeff_matrix finished')

    # sol1+sol2 is (qdd1, qdd2, ...)
    sol1 = qdd_coeff_matrix_inv * qdd_const_vector1
    sol2 = qdd_coeff_matrix_inv * qdd_const_vector2

    # defining fx and gx :
    fx = sm.zeros(2 * len_q, 1)
    gx = sm.zeros(2 * len_q, 1)

    for i in range(len_q):
        fx[i] = qdot[i]

    for i in range(len_q + 1, 2 * len_q):
        indx = i - (len_q + 1)
        fx[i] = sol1[indx]
        gx[i] = sol2[indx]

    gx[len_q] = 1

    return fx, gx




def linearize_state_equ(fx,
                        gx,
                        dynamics_symbs,
                        operation_point=None,
                        output_mode='numpy_array'):
    '''generates Linearzied form of state Equations at a given 
       equilibrium point.
    
    
         xdot= fx + gx.u ---> x_dot= A.delta_x + B.delta_u
     where :
              A= dfx/dxx|eq.Point + dgx/dxx.u|eq.Point 
              B= dg/du|eq.Point
                     
                     dim(A) : (n,n)
                     dim(B) : (n, 1)
    
    =========================================================
     INPUTS :
     -dynamics_symbs : a list containg q , qdot and u--> 
                        [q0, q1, qdot0, qdot1 , u]

     -operation_point    : a list containing --> [x0 , u0]  
                    

    ===========================================================
     OUTPUTS:
     - By default "numpy_array" A , B
     - if output_mode = "numpy_func"
       it returns A , B  as numpy functions of operating point 
     
    '''
    # finding q , qdot and u
    u = dynamics_symbs[-1]
    len_q = int((len(dynamics_symbs) - 1) / 2)
    q = dynamics_symbs[0:len_q]
    qdot = dynamics_symbs[len_q:2 * len_q]

    # calculating jacobians
    xx = sm.Matrix.vstack(sm.Matrix(q), sm.Matrix(qdot))
    df_dxx = fx.jacobian(xx)
    dg_dxx = gx.jacobian(xx)

    if output_mode == 'numpy_array':

        values = zip(dynamics_symbs, operation_point)
        values_dict = dict(values)

        # inserting operation point !
        A = (df_dxx + dg_dxx * u).subs(values_dict)
        B = gx.subs(values_dict)

        # converting to numpy
        A_numpy = np.array(A.tolist()).astype(np.float64)
        B_numpy = np.array(B.tolist()).astype(np.float64)
        ret = A_numpy, B_numpy

    elif output_mode == 'numpy_func':

        A = df_dxx + dg_dxx * u
        B = gx
        A_func = sm.lambdify(dynamics_symbs, A, 'numpy')
        B_func = sm.lambdify(dynamics_symbs, B, 'numpy')

        ret = A_func, B_func

    return ret


def lqr(A, B, Q, R, additional_outputs=False):
    """
    solves the continous time lqr controller:
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


def convert_qdd_to_func(fx, gx, dynamic_symbs, param_dict=None):
    '''
    convert state equations form sympy to functions
    output is a list containing [qdd0, qdd1, qdd2, ... ]

    '''
    u = dynamic_symbs[-1]
    len_q = int((len(dynamic_symbs) - 1) / 2)

    qdd_expr = fx + gx * u

    #substituting parameters in qdd_expr
    if isinstance(param_dict, dict):
        qdd_expr = qdd_expr.subs(param_dict)

    #converting qdd_expr to sympy function !
    qdd_func = [
        sm.lambdify(dynamic_symbs, qdd_expr[i + len_q], 'sympy')
        for i in range(len_q)
    ]

    return qdd_func


def pytraj_rhs(x, u, uref=None, t=None, pp=None):
    '''
     right hand side function for pytrajectory  
     YOU SHOULD: run convert_qdd_to_func once before using pytraj_rhs !!!
     
     GOOD TO KNOW :for pytrajectory rhs function must be defined as a 
                   "sympy" function and not "numpy" !
     
    '''
    qq = x
    u0, = u
    qdd_functions = cfg.pendata.trajectory.qdd_functions

    #frist defining xd[0 to len(q)] as qdots / ATTENTION: len(qq) = 2*len(q)
    q_len = int(len(qq) / 2)
    xd = [x[i + q_len] for i in range(q_len)]

    # xd[len(q)+1 to 2*len(q)] := qddts
    for i in range(q_len):
        #    xd.append(qdd_functions[i](*qq,u0))
        if len(qq) == 4:
            xd.append(qdd_functions[i](qq[0], qq[1], qq[2], qq[3], u0))
        elif len(qq) == 6:
            xd.append(qdd_functions[i](qq[0], qq[1], qq[2], qq[3], qq[4],
                                       qq[5], u0))
        elif len(qq) == 8:
            xd.append(qdd_functions[i](qq[0], qq[1], qq[2], qq[3], qq[4],
                                       qq[5], qq[6], qq[7], u0))

    ret = np.array(xd)

    return ret


# riccuti differential equation
def riccati_diff_equ(P, t, A_func, B_func, Q, R, dynamic_symbs, traj_label):
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
    # logging.debug('P_new: %s \n \n', P)

    # logging.debug('Debuging Message from riccati_diff_equ')
    # logging.debug('---------------------------------------------------')

    # logging.debug('t : %f \n', t)
    # extra conditions just for the case that odeint use samples
    # outside of the our t=(0, end_time)
    # print('riccati t: ', t)
    len_q = int((len(dynamic_symbs) - 1) / 2)
    cs_ret = cfg.pendata.trajectory.parallel_res[traj_label]

    if t < 0:
        x0 = [0] + [np.pi
                    for i in range(len_q - 1)] + [0.0 for i in range(len_q)]
        u0 = [0]
    else:
        x0 = cs_ret[0](t)
        u0 = cs_ret[1](t)

    # logging.debug('x0 : %s', x0)
    # logging.debug('u0 : %s \n', u0)

    equilib_point = np.hstack((x0, u0))
    A = A_func(*equilib_point)
    B = B_func(*equilib_point)

    # converting sympy.Matrix to numpy.array
    A = np.array(A.tolist()).astype(np.float64)
    B = np.array(B.tolist()).astype(np.float64)

    # logging.debug('P_current: %s', P)

    P = P.reshape((2 * len_q, 2 * len_q))
    P_dot = -P.dot(A) - (A.T).dot(P) + P.dot(B.dot(np_inv(R) *
                                                   (B.T).dot(P))) - Q

    ret = P_dot.reshape((2 * len_q)**2)
    # logging.debug('p_dot: %s', ret)
    return ret


# lqr_tracking function :
def generate_gain_matrix(R, B_func, P, Vect, dynamic_symbs, traj_label):
    '''
    returns input 'u' for tracking 
    each row of K_matrix include gain k at time t
     -->  num_rows= len(Vect), num_columns= len(states)
                 
    '''
    cs_ret = cfg.pendata.trajectory.parallel_res[traj_label]
    # K_matrix is a m*n matrix with m=len(Vect) and n=len_states :
    len_states = len(dynamic_symbs) - 1
    K_matrix = np.zeros((len(Vect), len_states))

    for i, t in enumerate(Vect):
        P_eq = P[i, :].reshape(len_states, len_states)

        # evaluating  B_equi at equilibrium point
        x0 = cs_ret[0](t)
        u0 = cs_ret[1](t)
        operating_point = np.hstack((x0, u0))

        B = B_func(*operating_point)
        # converting B (sympy.Matrix) to numpy.array
        B = np.array(B.tolist()).astype(np.float64)

        # finding k and add it to K_matrix
        K_matrix[i, :] = np_inv(R) * B.T.dot(P_eq)


    return K_matrix


# converting symbolic state equations to functions
def sympy_states_to_func(fx, gx):
    '''it converts symbolic state equations to functions

    ATTENTION :
       - the difference between this and qdd_to_func is
         that qdd_to_func returns 'sympy' funcitons, 
         sympy_states_to_func on the other hand returns
         numpy functions !
       -it uses fx, gx and parameter_values stored in config.py
        fx and gx are calculated in sys_model.py and stored in 
        Pen_Container
    ============================================================

    INPUTS:
    - cfg.pendata 

    OUTPUTS :
    - callable state_func(state, input) 
    '''

    dynamic_symbs = cfg.pendata.model.dynamic_symbs

    u = dynamic_symbs[-1]
    num_states = len(dynamic_symbs) - 1

    # logging.debug('u : %s', u)
    # logging.debug('fx_expr: %s', fx)
    # logging.debug('gx_expr: %s', gx)

    xdot_expr = fx + gx * u
    # logging.debug('xdot_exr: %s', xdot_expr)

    xdot_func = [
        sm.lambdify(dynamic_symbs, xdot_expr[i], modules='numpy')
        for i in range(num_states)
    ]

    def state_func(state, inputs):
        '''
        given state and inputs it returns x_dot
        '''
        u, = inputs
        x = state
        len_states = len(x)
        if len_states == 4:
            xd = [
                xdot_func[i](x[0], x[1], x[2], x[3], u)
                for i in range(len_states)
            ]
        elif len_states == 6:
            xd = [
                xdot_func[i](x[0], x[1], x[2], x[3], x[4], x[5], u)
                for i in range(len_states)
            ]
        elif len_states == 8:
            xd = [
                xdot_func[i](x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], u)
                for i in range(len_states)
            ]

        x_dot = np.array(xd)

        return x_dot

    return state_func


# ode_function to be used in odeint
def ode_function(x,
                 t,
                 xdot_func,
                 K_matrix,
                 Vect,
                 traj_label,
                 mode='Closed_loop'):
    '''
    it's the dx/dt=func(x, t, args)  to be used in odeint
    (the first two arguments are system state x and time t)

    there are two modes available:
     - Closed_loop is defualt and can be used for tracking
     - Open_loop could be activated by setting  mode='Open_loop'
    

    ATTENTION :
      - use sympy_states_to_func to produce xdot functions from  
        sympy expresisons. 
        (you have to run sympy_state_to_func once and store the result
        so you could pass it as xdot_func )

    '''
    # logging.debug('x_new: %s \n \n', x)
    # logging.debug('Debugging Message from ode_function')
    # logging.debug(
    #     '----------------------------------------------------------------')
    # n=len(Vect)
    # logging.debug('t: %s \n', t)

    cs_ret = cfg.pendata.trajectory.parallel_res[traj_label]
    # print('t ode_func', t)
    if t > Vect[-1]:
        t = Vect[-1]

    xs = cs_ret[0](t)
    us = cs_ret[1](t)
    sys_dim = len(xs)
    if mode == 'Closed_loop':
        k_list = [np.interp(t, Vect, K_matrix[:, i]) for i in range(sys_dim)]
        k = np.array(k_list)
        delta_x = x - xs
        delta_u = (-1) * k.T.dot(delta_x)
        inputs = us + delta_u
        # loggings :
        # logging.debug('k :%s', k)
        # logging.debug('delta_x: %s', delta_x)
        # logging.debug('delta_u: %s \n', delta_u)

    elif mode == 'Open_loop':
        inputs = us
    
    if inputs > 40:
        # ipydex.IPS()
        inputs=[40]
        # print(inputs)

    elif inputs < -40:
        # ipydex.IPS()
        inputs=[-40]    
        # print(inputs)

    state = x
    # logging.debug('us: %s', us)
    # logging.debug('xs:%s \n', xs)
    # logging.debug('state: %s', state)
    # logging.debug('inputs: %s \n', inputs)

    xdot = xdot_func(state, inputs)
    # logging.debug('x_current: %s', x)
    # logging.debug('xdot : %s ', xdot)

    return xdot


def calculate_u_cl(ct, x_cl, k_matrix, tvec, traj_label, extraOutputs=False):
    ''' calculates u_cl from x_cl
        deltax = x-xs
        delta_u = (-1) * k.T.dot(delta_x)
        u_cl = us + delta_u
    '''
    cs_ret = cfg.pendata.trajectory.parallel_res[traj_label]

    u_cl= []
    deltaU=[]
    deltaX=[]
    for idx, x in enumerate(x_cl) :

        t=tvec[idx]
        xs = cs_ret[0](t)
        us = cs_ret[1](t)
        k= k_matrix[idx, :]

        delta_x = x - xs
        delta_u = (-1) * k.T.dot(delta_x)
        u = us + delta_u
        if u > 40:
            u=[40]
            # print(u)

        elif u < -40:
            # ipydex.IPS()
            u=[-40]    
            # print(u)
        u_cl.append(u)
        deltaU.append(delta_u)
        deltaX.append(delta_x)

    deltaU= np.array(deltaU)
    deltaX= np.array(deltaX)
    u_cl=np.array(u_cl)
    # ct.tracking.tracking_res[traj_label].update({'ucl': np.array(u_cl)})
    if extraOutputs:
        ret=u_cl, deltaU, deltaX
    else:
        ret= u_cl
    return ret



def generate_position_vector_func(In_frame, origin, point, dynamic_variables):
    ''' retuns the position of the point on reference_frame
     in inertial reference frame (In_frame) as a fucntion
     of dynamic_variables 
     '''
    # finding the vector of point  in respect to In_frame
    point_vector = point.pos_from(origin)
    point_vector_x = point_vector.dot(In_frame.x)
    point_vector_y = point_vector.dot(In_frame.y)

    # substituting dummy variables instead of dynamic_variables
    # couse there are funcitons of times and they couldn't be unpacked
    # like a normal sm.symbols as args for the funtions
    dummy_symbols = [sm.Dummy() for i in dynamic_variables]
    dummy_dict = dict(zip(dynamic_variables, dummy_symbols))

    point_vector_dummy = [
        point_vector_x.subs(dummy_dict),
        point_vector_y.subs(dummy_dict)
    ]

    # lamdify the point vector
    point_vector_func = [
        sm.lambdify(dummy_symbols, point_vector_component, modules='numpy')
        for point_vector_component in point_vector_dummy
    ]
    return point_vector_func


def generate_transformation_matrix_2d(In_frame, origin, reference_frame,
                                      point):
    '''shamplessly modified version of pydy's 3D transfromation !
    Generate a symbolic 2D transformation matrix, with respect to the 
    reference frame and point.
    
    INPUTS:
     - In_frame : Inertial reference frame
     - reference_frame : A frame with respect to whicht transformation matrix 
       is generated
     - Point : A point with respect to which transformation is generated   
    '''
    rotation_matrix = In_frame.dcm(reference_frame)
    rotation_matrix_2d = sm.Matrix(
        [rotation_matrix[i, j] for i in range(2) for j in range(2)]).reshape(
            2, 2)
    trans_matrix = sm.Identity(3).as_mutable()
    trans_matrix[:2, :2] = rotation_matrix_2d

    # defining the vector from origin to the point
    point_vector = origin.pos_from(point)

    # calculating translation part of Transformation Matrix
    trans_matrix[2, 0] = point_vector.dot(reference_frame.x)
    trans_matrix[2, 1] = point_vector.dot(reference_frame.y)
    return trans_matrix


def generate_transformation_matrix_func(trans_matrix, dynamic_variables):
    ''' returns a a list of functions which computes the numerical values of 
    the transformation_matrix_2d
    
    '''
    dummy_symbols = [sm.Dummy() for i in dynamic_variables]
    dummy_dict = dict(zip(dynamic_variables, dummy_symbols))

    # reshaping the matrix to use lambdify --> reshape because
    # lambdify returns null instead of null vector or matrxi !
    transfrom = trans_matrix.subs(dummy_dict).reshape(9, 1)
    trans_funcs = []
    for i in range(9):
        trans = trans_matrix[i]
        func = sm.lambdify(dummy_symbols, trans, modules='numpy')
        trans_funcs.append(func)
    return trans_funcs


def array_compare(a, b, tol=1e-6):
    ''''difference between two np.arrays a and b
    TODO: should rewite this!
    - Default tol is 1e-6
    ===========
    - Returns :
             a tupel consist of differnece array 
             and maximal difference
    '''
    func1 = lambda x: x if abs(x) >= tol else 0
    func2 = lambda x, y: np.array([func1(xi - yi) for xi, yi in zip(x, y)])
    delta = np.array([func2(xs1, xs2) for xs1, xs2 in zip(a, b)])
    max_diff = np.amax(np.array([func1(deltai) for deltai in delta]))

    return delta, max_diff



def parallelized_model_generator(param_tol_dicts, pool_size=4):
    '''multiple model generator at the same time :
       param_tol_dicts : a list containing param_tol_dict 

    '''
    processor_pool = Pool(pool_size)
    processor_pool.map(generate_sys_model_with_parameter_deviation , param_tol_dicts )

    return




def generate_sys_model_with_parameter_deviation(param_tol_dict=None, default_tol=0.04):
    '''system modeling with parameter deviation desired!
       
       ------
       INPUTS:
       param_tol_dict : dict , a dictionary with the name of the parameters as keys
       and desired tolerance as values, if not given it uses default_tol for every 
       parameters
    '''
    from sys_model import system_model_generator
    ct=cfg.pendata
    label=ct.label

    param_values = ct.parameter_values
    param_keys = [
        'l1', 'l2', 'l3', 'a1', 'a2', 'a3', 'm0', 'm1', 'm2', 'm3', 'J0', 'J1',
        'J2', 'J3', 'd1', 'd2', 'd3', 'g', 'f'
    ]
    param_dict = dict(zip(param_keys, param_values))

    flag=True
    if not isinstance(param_tol_dict, dict):
        flag = False
        param_tol_dict = dict([(param_key, default_tol if param_key is not 'g'
                                and param_key is not 'f'
                                and param_key is not 'J0' else 0)
                               for param_key in param_keys])
        # updagte tol_dict in Container:                       
        cfg.pendata.parameter_tol_dicts= param_tol_dict                       

    keys, values = zip(*param_tol_dict.items())
    # ipydex.IPS()
    for key in keys:

        new_value = param_dict[key] *(param_tol_dict[key] + 1)
        param_dict.update({key: new_value})

    if flag == True :
        model_name = ""
        keys, values= zip(*param_tol_dict.items())
        for key in keys :
            model_name += str(key)
        model_name=model_name+'_'
        for value in values :
            model_name += str(value)


    else :
        model_name= 'defaultTol_'+ str(default_tol)

    model_file_name= 'sys_model_' + 'with_param_deviation_'+ model_name +'_'+ label + '.pkl'


    # ct.model.param_with_deviation = param_with_deviation
    ct.model.param_dict_with_deviation = param_dict
    ct.model.param_tol_dict = param_tol_dict
    ct.model.default_tol = str(default_tol) if flag == False else 'variable'
    ct.model.model_file_name= model_file_name
    system_model_generator(ct, param_deviation=True)

    return

def system_model_generator_original(ct):
    """generates model with original parameter
    """
    from sys_model import system_model_generator

    label=ct.label    
    model_file_name= 'sys_model_' + 'without_param_deviation_'+'_'+ label + '.pkl'
    system_model_generator(ct, param_deviation=False)

    return
    



def dump_model(model_file_name,data):
    ''' dumps model in a folder named: sys_models
    
    '''
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]
    if current_dir == 'core':
        os.chdir('..')
        createFolder('sys_models')
        os.chdir('sys_models')
    elif current_dir == 'triple_pendulum':
        createFolder('sys_models')
        os.chdir('sys_models')

    with open(model_file_name, 'wb') as file:
        dill.dump(data, file)
        file.close()
    # back to the folder we started form :
    os.chdir(current_path)

    return

def generate_model_names_from_tol_dict(param_tol_dicts):
    ''' generates model_names from a given tol_dicts
    '''
    model_names=[]
    for param_tol_dict in param_tol_dicts:
        keys, values = zip(*param_tol_dict.items())
        model_name = ""
        for key in keys :
            model_name += str(key)
        model_name += '_'
        for value in values :
            model_name += str(value)
        model_names.append(model_name)

    return model_names



def load_sys_model(ct, model_without_param_deviation=None, deviation_percentage=None, param_tol_dicts=None):
    '''Loading system model from pickle file and 
       save it in pendata.model
       it also updates model_list with each model loaded !
    ***ATTENTION:
        frames could not be loaded ! therefore for visualization 
        you should still run system_model_generator
    '''
    label=ct.label
    if isinstance(param_tol_dicts, list):
        model_file_names=[]
        model_names=[]
        for param_tol_dict in param_tol_dicts:
            keys, values = zip(*param_tol_dict.items())
            model_name = ""
            for key in keys :
                model_name += str(key)
            model_name += '_'
            for value in values :
                model_name += str(value)
            model_names.append(model_name)
            model_file_name= 'sys_model_' + 'with_param_deviation_'+ model_name +'_'+ label + '.pkl'
            model_file_names.append(model_file_name)

    elif deviation_percentage :
        model_name= 'defaultTol_'+ str(deviation_percentage)
        model_file_names= ['sys_model_' + 'with_param_deviation_'+ model_name +'_'+ label + '.pkl']

    elif model_without_param_deviation :
        model_file_names=['sys_model_' + 'without_param_deviation_'+ label + '.pkl']

    # go to the directory sys_models and load files
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]
    if current_dir == 'core':
        os.chdir('..')
        os.chdir('sys_models')
    elif current_dir == 'triple_pendulum':
        os.chdir('sys_models')

    # check if model is already exist !

    if not hasattr(ct.model, 'models_with_tol_dicts'):
        ct.model.models_with_tol_dicts={}

    if not hasattr(ct.model, 'models_with_default_tol'):
        ct.model.models_with_default_tol={}


    if not  ('model_list' in ct.model.models_with_tol_dicts) :
        ct.model.models_with_default_tol.update({'model_list':[]})

    if not  ('model_list' in ct.model.models_with_tol_dicts):
        ct.model.models_with_tol_dicts.update({'model_list':[]})

    # loop throught the file names and load files :

    count=0
    for data_name in model_file_names:

        with open(data_name, 'rb') as file:
            sys_model = dill.load(file)

        if model_without_param_deviation:
            ct.model.param_dict = sys_model['param_dict']
            ct.model.fx = sys_model['fx']
            ct.model.gx = sys_model['gx']
            ct.model.model_names_list= ['original']
            print('model without param deviation is stored')

        elif deviation_percentage:

            key= 'defaultTol_'+ str(deviation_percentage)
            value= dict([('param_dict', sys_model['param_dict']),
                      ('fx', sys_model['fx']), ('gx', sys_model['gx']),
                      ('default_tol', sys_model['default_tol'])])


            ct.model.models_with_default_tol.update({key: value})
            ct.model.models_with_default_tol['model_list'].append(key)
            print('model with default tol is stored')


        elif param_tol_dicts :
            temp= data_name.split("_")
            model_name= temp[5] + "_" + temp[6]
            key= model_name
            value= dict([('param_dict', sys_model['param_dict']),
                      ('fx', sys_model['fx']), ('gx', sys_model['gx']),
                      ('param_tol_dict', sys_model['param_tol_dict'])])
            ct.model.models_with_tol_dicts.update({key: value})
            ct.model.models_with_tol_dicts['model_list'].append(key)
            count += 1


    ct.model.q= sys_model['q']
    ct.model.qdot= sys_model['qdot']
    ct.model.qdd= sys_model['qdd']
    ct.model.dynamic_symbs= sys_model['dynamic_symbs']

    print('{} models with tol_dict is stored'.format(count))
    # go back  to the root !
    os.chdir(current_path)

    return






def load_traj_splines(ct, pfname,traj_lable_list=None):
    '''Loades splines from .pkl file containing the reults
    ParallelizedTP and saves them to ct.trajectory.parallel_res  
    you may list your desired trajectories in traj_label list ,
    if you want to convert a specific trajectory to traj_splines
    (it saves time not to convert all the trajectory results !) 
     
    '''
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]

    if current_dir == 'core':
        os.chdir('..')
        os.chdir('traj_results')

    elif current_dir == 'triple_pendulum':
        os.chdir('traj_results')


    with open(pfname, 'rb') as pfile:
        trajectories = dill.load(pfile)

    if isinstance(traj_lable_list, list):
        temp=dict(trajectories)
        traj_labels, values=zip(*temp.items())
        desired_trajectories=[]
        for traj_label in traj_labels:
            for desired_traj_label in traj_lable_list :
                if traj_label == desired_traj_label :
                    traj= temp[traj_label]
                    desired_trajectories.append((traj_label, traj))
        trajectories=desired_trajectories


    parallel_res = []
    for trajectory in trajectories:
        cont_dict = trajectory[1][1]
        traj_label = trajectory[0]
        traj_spline = aux.unpack_splines_from_containerdict(cont_dict)
        xxf, uuf = aux.get_xx_uu_funcs_from_containerdict(cont_dict)
        parallel_res.append((traj_label, (xxf, uuf)))

    if not hasattr(ct.trajectory, 'parallel_res') :
        ct.trajectory.parallel_res = {}

    ct.trajectory.parallel_res.update(dict(parallel_res))
    print("Trajectories Written to ct.trajectory.parallel_res")
    os.chdir(current_path)

    return



def load_traj_from_numpyFile(ct, uName, xName):
    """ load trajectory results form numpy file and save 
        as functions to be used in tracking

    """
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]

    if current_dir == 'core':
        os.chdir('..')
        os.chdir('pytraj_results')

    elif current_dir == 'triple_pendulum':
        os.chdir('pytraj_results')
    

    u_traj= np.load(uName)
    u_traj.reshape((1000,1))
    x_traj= np.load(xName)

    def x_res(t):

        tt=np.linspace(0,2,1000)
        if t < 0:
            x_ret=x_traj[0]
        elif t > 2:
            x_ret=x_traj[-1]
        
        else:
            x_list=[np.interp(t, tt, x_traj[:,i]) for i in range(8)]
            x_ret=np.array(x_list)
        
        return x_ret    

    def u_res(t):
        
        tt=np.linspace(0,2,1000)
        if t < 0:
            u_ret=u_traj[0]
        elif t > 2:
            u_ret=u_traj[-1]
        
        else:
            u_ret=np.interp(t, tt, u_traj[:,0])
        
        return u_ret
    traj_label='pyTraj2'    
    parallel_res=[(traj_label,(x_res, u_res))]
    if not hasattr(ct.trajectory, 'parallel_res') :
        ct.trajectory.parallel_res = {}

    ct.trajectory.parallel_res.update(dict(parallel_res))
    print("Trajectories Written to ct.trajectory.parallel_res")
    os.chdir(current_path)
    
    return


   

def load_pytrajectory_results(ct, pfname):
    '''  if you want to pickle the results again you should'nt
         use load_traj_splines! (for example: multiprocessing needs
         arguments to be pickleable!)
         Instead you can use load_pytrajectory_results
    '''
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]

    if current_dir == 'core':
        os.chdir('..')
        os.chdir('traj_results')

    elif current_dir == 'triple_pendulum':
        os.chdir('traj_results')

    with open(pfname, 'rb') as pfile:
        pytrajectory_res = dill.load(pfile)

    if not hasattr(ct.trajectory, 'pytrajectory_res'):
        ct.trajectory.pytrajectory_res= {}


    ct.trajectory.pytrajectory_res.update(dict(pytrajectory_res))
    print("pytrajectory_results are loaded to ct.trajectory.pytrajectory_res")
    os.chdir(current_path)

    return


def convert_container_res_to_splines(ct, traj_label, model_key):
    ''' Converts containerized results of pytrajectory 
        to callabel functions of xx and uu

        INPUTS:
        ct: container = cfg.pendata
        traj_label : label of the desired trajectory 

        OUTPUTS:
        parallel_res : will be saved in ct.trajectory.parallel_res
    '''
    trajectory = ct.trajectory.pytrajectory_res[traj_label]
    cont_dict = trajectory[1]
    # traj_spline = aux.unpack_splines_from_containerdict(cont_dict)
    xxf, uuf = aux.get_xx_uu_funcs_from_containerdict(cont_dict)
    ct.trajectory.parallel_res[traj_label] = (xxf, uuf)
    print("{} - {} : Trajectories Written to ct.trajectory.parallel_res".format(traj_label,model_key))


def load_tracking_results(ct, file_names_dict):
    '''loads x_closed_loop , u_closed_loop, k_matrix ...
    results will be saved in ct.tracking.tracking_res['traj_label']['x_closed_loop']
    (a dictionary with "seed_time"  as keys, and values are dictionaries with keys :
    x_closed_loop, u_closed_loop and so an )

    INPUTS : 
    
    ct: container of type Pendata
    file_names_dict: 
    
    TODO: 
    '''
    # provide a dictionary for the results to be saved to  :

    # go to the direcotory where the files are :
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]
    if current_dir == 'core':
        os.chdir('..')
        os.chdir('tracking_results')
    elif current_dir == 'triple_pendulum':
        os.chdir('tracking_results')

    seed_times, file_names = zip(*file_names_dict.items())

    ct.tracking.tracking_res = dict(
        [(seed_time, {}) for seed_time in seed_times])

    for seed_time in seed_times:

        for file_label, file_name in file_names_dict[seed_time].iteritems():
            directory = seed_time
            key = file_label
            value = np.load(os.path.join(directory, file_name))
            ct.tracking.tracking_res[seed_time].update({key: value})

    # return to the directory where we began !
    os.chdir(current_path)
    return


def save_tracking_results_to_contianer(ct, traj_label, x_closed_loop):
    ct.tracking.tracking_res[traj_label]['x_closed_loop'] = x_closed_loop
    return


def createFolder(directory):
    '''creats a subfolder in a folder your run the program
       with the name directory
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def np_save(directory, file_name, data):
    ''' saves the data(numpy.array) in a folder specified with
        directory and with the name file_name
        directory will be created so you should just specify 
        a name for it !  if it already exists no new folder will be 
        created!
        ======= 
        INPUTS:
        directory : string ,  name of the directory to be created
        file_name : string ,  name of the file
        data :      np.array, your data to be saved   
    '''
    #create a folder for file to be saved to:
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]
    if current_dir == 'core':
        os.chdir('..')
        createFolder('tracking_results')
        os.chdir('tracking_results')
    elif current_dir == 'triple_pendulum':
        createFolder('tracking_results')
        os.chdir('tracking_results')

    createFolder(directory)

    # save data to a file
    np.save(os.path.join(directory, file_name), data)

    # back to the directory where we started !
    os.chdir(current_path)
    return




def read_tracking_results_from_directories(directories):
    '''generates a list of all numpy(.npy) files in a directory specified
       and save the files in dictionary with NESTED keywords:
       tracking_res: 
                  [directory]
                  [res_type]
                  [deviation_type]
                  [model_key]
                  [tx_key]

       it spects that the directories are in triple_pendulum/tracking_results
    INPUTS:
    directories: list of  strings , names of the directories
    
    '''
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]
    if current_dir == 'core':
        os.chdir('..')
        os.chdir('tracking_results')
    elif current_dir == 'triple_pendulum':
        os.chdir('tracking_results')

# deltaU_deviation_None_original_00_t0_0_x0_0_QR_1.01.0_Inverted_triple_Pendulum_mat50_2.2.npy
# x_closed_loop_deviation_None_original_00_t0_0_x0_0_QR_1.01.0_Inverted_triple_Pendulum_mat50_2.2.npy
    tracking_res= dict([(directory, {}) for directory in directories])

    for directory in directories:
        os.chdir(directory)


        for file_name in glob.glob("*.npy"):
            tmp= file_name.split('_')

            if tmp[0] == 'simulation':
                res_type='simulation_time'
                value = np.load(file_name)
                tracking_res[directory].update({res_type: value})

            else :
                if tmp[0] == 'K':
                    res_type= 'k_matrix'
                    deviation_type= tmp[3]
                    model_key= tmp[4]+'_'+tmp[5]
                    tx_key = tmp[7]+tmp[9]
                    qr_key = tmp[11]

                elif tmp[0] == 'x' :
                    res_type= 'x_cl'
                    deviation_type= tmp[4]
                    model_key= tmp[5]+'_'+tmp[6]
                    tx_key = tmp[8]+tmp[10]
                    qr_key = tmp[12]

                elif tmp[0] == 'u' :
                    res_type= 'u_cl'
                    deviation_type= tmp[4]
                    model_key= tmp[5]+'_'+tmp[6]
                    tx_key = tmp[8]+tmp[10]
                    qr_key = tmp[12]

                elif tmp[0] == 'deltaX':
                    res_type = 'deltaX'
                    deviation_type= tmp[2]
                    model_key= tmp[3]+'_'+tmp[4]
                    tx_key = tmp[6]+tmp[8]
                    qr_key = tmp[10]

                elif tmp[0] == 'deltaU':
                    res_type = 'deltaU'
                    deviation_type= tmp[2]
                    model_key= tmp[3]+'_'+tmp[4]
                    tx_key = tmp[6]+tmp[8]
                    qr_key = tmp[10]    

                if not (res_type in tracking_res[directory] ):
                    tracking_res[directory].update({res_type:{}})
                if not (deviation_type in tracking_res[directory][res_type]):
                    tracking_res[directory][res_type].update({deviation_type: {}})
                if not ( model_key in tracking_res[directory][res_type][deviation_type]):
                    tracking_res[directory][res_type][deviation_type].update({model_key :{}})
                if not (tx_key in tracking_res[directory][res_type][deviation_type][model_key]):
                    tracking_res[directory][res_type][deviation_type][model_key].update({tx_key :{}})
                    

                value= np.load(file_name)
                tracking_res[directory][res_type][deviation_type][model_key][tx_key].update({qr_key:value})



        # go back and search for files in another direcotry
        os.chdir('..')

    #go back to the directory where we began
    os.chdir(current_path)

    return tracking_res




def generate_trajectory_name(types, traj_labels, number_of_splines_list,  constraints):
    '''
    type                      string:    'swingup_splines', ...
    traj_lables               list  :     containing of 'seed_time'
    number_of_splines_list:   list  :     number of splines for each trajectory according
                                          to traj_labels
    constraints               dict  :     constraints dict                  
    '''
    seed_time_label=""
    for traj_label in traj_labels:
        seed_time_label+= "_"+ traj_label

    spline_label= "_splines"
    for num_of_splines in number_of_splines_list :
        spline_label += '_' + str(num_of_splines)

    con_label="_"
    for key, value in constraints.iteritems() :
        con_label+= key + "_"+ str(value)+"_"

    pfname= types + '_' + '_add_infos_' + seed_time_label+ spline_label + con_label+'.pcl'

    return pfname


def dump_trajectory_results(types, traj_labels, number_of_splines_list,  constraints, trajectories):

    ''' generates a name for trajecotry and dump it to traj_results in triple_pendulum
       
    '''
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]

    if current_dir == 'core':
        os.chdir('..')
        createFolder('traj_results')
        os.chdir('traj_results')

    elif current_dir == 'triple_pendulum':
        createFolder('traj_results')
        os.chdir('traj_results')

    pfname= generate_trajectory_name(types, traj_labels, number_of_splines_list, constraints)

    with open(pfname, 'wb') as pfile:
        dill.dump(trajectories, pfile)
        pfile.close()


    print("Trajectories are dumped to {}".format(os.path.join('traj_results',pfname)))

    os.chdir(current_path)

    return

def split_trajectory_results(ct, pfname):
    '''load results of pytrajectory and split them into 
       seperate results and dump them again !
       
    '''
    load_pytrajectory_results(ct, pfname)
    traj_res= ct.trajectory.pytrajectory_res
    types= 'swingup_splines'

    for key, value in traj_res.iteritems():
        traj_labels=[key]
        num_of_splines_list= [value[1]['x1'].n]
        constraints={'x0':None, 'x4':None}
        trajectory=[(key, value)]
        dump_trajectory_results(types, traj_labels, num_of_splines_list, constraints, trajectory )

    return
