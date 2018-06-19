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

import cfg

import ipydex

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

    xx = sm.Matrix([q+qdot+[u]])
    # we just need qdds so:
    fxu = sm.Matrix((fx + gx * u))
    jac = fxu.jacobian(xx)
    
    # to avoid lambdify's bug we need to lambdify every single component
    jac_funcs = [
        sm.lambdify(xx, jac[i, j], 'numpy') for i in range(2*len(q))
        for j in range(2*len(q)+1)
    ]

    def jac_func(x, u):
        ''' returns numpy array of jacobian evaluated at x and u
            Inputs :x and u should be given as numpy array
        '''
        n=int(len(x))
        xx=np.array( x.tolist() + u.tolist())
        jac= np.array([jac_funcs[i](*xx) for i in range(n*(n+1))]).reshape(n, n+1)

        return jac

    return jac_func

                
    

    



def generate_state_equ_new(mass_matrix, forcing_vector, qdot, qdd, u):
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


def generate_state_equ(mass_matrix, forcing_vector, qdot, qdd, u):
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
    # --->  qdd_coeff_matrix * qdd_vect = qdd_const_vector

    sol_qdd = qdd_coeff_matrix.inv() * qdd_const_vector
    print('qdd is ready!')
    # sometimes collec() dosent return wrong results without expanding
    # the expressions first ! so we have to expand :-D
    sol_list = [sol_qdd[i].expand() for i in range(len_q - 1)]

    # finding terms with and without input u to determine fx and gx
    collected_qdd = [
        sm.collect(sol_list[i], u, evaluate=False) for i in range(len_q - 1)
    ]

    fx = sm.zeros(2 * len_q, 1)
    gx = sm.zeros(2 * len_q, 1)

    for i in range(len_q):
        fx[i] = qdot[i]

    for i in range(len_q + 1, 2 * len_q):
        indx = i - (len_q + 1)
        fx[i] = collected_qdd[indx][1]
        gx[i] = collected_qdd[indx][u]

    gx[len_q] = 1

    fx = sm.ImmutableMatrix(fx).simplify()
    gx = sm.ImmutableMatrix(gx).simplify()
    return fx, gx


def generate_state_equ_old(mass_matrix, forcing_vector, qdot, qdd, u):
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
    expr_list = [expr[i] for i in range(len(qdd) + 1, 2 * len(qdd))]

    var_list = [qdd[i] for i in range(1, len(qdd))]
    print('variables and expressions are ready')
    sol = solve(expr_list, var_list)
    print('solving for qddots is finished!')
    #determining fx and gx
    qdd_expr = [sol[qdd[i]].expand() for i in range(1, len(qdd))]

    #finding terms with and without input term 'u' to determine fx and gx
    collected_qdd = [
        sm.collect(qdd_expr[i], u, evaluate=False)
        for i in range(len(qdd) - 1)
    ]
    print('collecting terms is finished')
    fx = sm.zeros(len(xx_dot), 1)
    gx = sm.zeros(len(xx_dot), 1)
    print('fx and gx are ready')
    for i in range(len(qdot)):
        fx[i] = qdot[i]
        gx[i] = 0.0

    for i in range(len(qdot) + 1, len(xx_dot)):
        indx = i - (len(qdot) + 1)
        fx[i] = collected_qdd[indx][1]
        gx[i] = collected_qdd[indx][u]
    gx[len(qdd)] = 1
    return fx, gx


def generate_state_equ_test(mass_matrix, forcing_vector, qdot, qdd, u):
    '''
    given mass_matrix and forcing_vector of a Kane's Method it
    generates state equation :
                         xdot=f(x)+g(x).u
               
                  'ATTENTION : it assumes qdd0 as an Input u'
    
    '''

    xdot = mass_matrix.inv() * forcing_vector
    fx = sm.Matrix([xdot[i] for i in range(2 * len(qdot))])
    fx[len(qdot)] = 0
    gx = sm.zeros(2 * len(qdot), 1)
    gx[len(qdot)] = 1

    return fx, gx


def linearize_state_equ(fx,
                        gx,
                        dynamics_symbs,
                        operation_point=None,
                        output_mode='numpy_array'):
    '''
    TODO: solve potential promblem with lambdify if the results are null

    generate Linearzied form of state Equations at a given 
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
        print('A_numpy')
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
     right hand side function for pytrajecotry  
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
def riccati_diff_equ(P, t, A_func, B_func, Q, R, dynamic_symbs):
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
    print('riccati t: ', t)
    len_q = int((len(dynamic_symbs) - 1) / 2)
    cs_ret = cfg.pendata.trajectory.cs_ret

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
def generate_gain_matrix(R, B_func, P, Vect, dynamic_symbs):
    '''
    returns input 'u' for tracking 
    each row of K_matrix include gain k at time t
     -->  num_rows= len(Vect), num_columns= len(states)
                 
    '''
    cs_ret = cfg.pendata.trajectory.cs_ret
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
def sympy_states_to_func():
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
    fx = cfg.pendata.model.fx
    gx = cfg.pendata.model.gx
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
def ode_function(x, t, xdot_func, K_matrix, Vect, mode='Closed_loop'):
    '''
    it's the dx/dt=func(x, t, args)  to be used in odeint
    (the first two arguments are system state x and time t)

    there are two modes available:
     - Closed_loop is defualt and can be used for tracking
     - Open_loop could be activated by setting  mode='Open_loop'
    

    ATTENTION :
      - use sympy_states_to_func to produce xdot functions out of 
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

    cs_ret = cfg.pendata.trajectory.cs_ret
    print('t ode_func', t)
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

    state = x

    # logging.debug('us: %s', us)
    # logging.debug('xs:%s \n', xs)
    # logging.debug('state: %s', state)
    # logging.debug('inputs: %s \n', inputs)

    xdot = xdot_func(state, inputs)
    # logging.debug('x_current: %s', x)
    # logging.debug('xdot : %s ', xdot)

    return xdot


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



    def array_compare(a,b, tol=1e-6):
        '''difference between two np.arrays a and b
       
       - Default tol is 1e-6
       ===========
       - Returns :
                a tupel consist of differnece array 
                and maximal difference
    '''
    func1= lambda x: x if abs(x) >= tol else  0
    func2= lambda x, y: np.array([func1(xi-yi) for xi,yi in zip(x,y) ]) 
    delta= np.array([func2(xs1,xs2) for xs1, xs2 in zip(a, b) ])
    max_diff= np.amax(np.array([func1(deltai) for deltai in delta]))
        
    
    return delta, max_diff