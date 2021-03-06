# coding: utf-8
#=============================================================
# Standard Python modules
#=============================================================
import sys, os
import dill
#=============================================================
# External Python modules
#=============================================================
from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=True)

import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
import mpmath as mp
import scipy as sc

import ipydex
#=============================================================
# other moduless
#=============================================================
from myfuncs import generate_state_equ
from myfuncs import dump_model

import cfg

#=============================================================
# Systme Model
#=============================================================


def system_model_generator(ct, param_deviation=False):
    '''
    Modeling inverted pendulum using Kane's Method

    '''


    # Defining symbolic Variables
    n = ct.number_of_pendulums
    q = me.dynamicsymbols('q:{}'.format(n + 1))  # generalized coordinates
    qdot = me.dynamicsymbols('qdot:{}'.format(n + 1))  #generalized speeds
    qdd = me.dynamicsymbols('qddot:{}'.format(n + 1))
    f = me.dynamicsymbols('f')
    u = sm.symbols('u')
    m = sm.symbols('m:{}'.format(n + 1))
    J = sm.symbols('J:{}'.format(n + 1))
    l = sm.symbols('l1:{}'.format(n + 1))  # lenght of each pendlum
    a = sm.symbols('a1:{}'.format(n + 1))  #location of Mass-centers
    d = sm.symbols('d1:{}'.format(n + 1))  #viscous damping coef.
    g, t = sm.symbols('g t')
    dynamic_symbs = q + qdot + [u]
    # intertial reference frame
    In_frame = me.ReferenceFrame('In_frame')
    # Origninal Point O in Reference Frame :
    O = me.Point('O')
    O.set_vel(In_frame, 0)
    # The Center of Mass Point on cart :
    C0 = me.Point('C0')
    C0.set_pos(O, q[0] * In_frame.x)
    C0.set_vel(In_frame, qdot[0] * In_frame.x)
    cart_inertia_dyadic = me.inertia(In_frame, 0, 0, J[0])
    cart_central_inertia = (cart_inertia_dyadic, C0)
    cart = me.RigidBody('Cart', C0, In_frame, m[0], cart_central_inertia)
    kindiffs = [q[0].diff(t) - qdot[0]]  # entforcing qdot=Omega
    frames = [In_frame]
    mass_centers = [C0]
    joint_centers = [C0]
    central_inertias = [cart_central_inertia]
    forces = [(C0, f * In_frame.x - m[0] * g * In_frame.y)]
    # torques = []
    # cart_potential = 1 / 3 * d[0] * qdot[1]**3
    # potentials = [cart_potential]
    # cart.potential_energy= cart_potential
    rigid_bodies = [cart]
    # Lagrangian0 = me.Lagrangian(In_frame, rigid_bodies[0])
    # Lagrangians=[Lagrangian0]
    for i in range(n):
        #Creating new reference frame
        Li = In_frame.orientnew('L' + str(i), 'Axis',
                                [ q[i + 1], In_frame.z])
        Li.set_ang_vel(In_frame, qdot[i + 1] * In_frame.z)
        frames.append(Li)
        # Creating new Points representing mass_centers
        Pi = joint_centers[-1].locatenew('a' + str(i + 1), a[i] * Li.y)
        Pi.v2pt_theory(joint_centers[-1], In_frame, Li)
        mass_centers.append(Pi)
        #Creating new Points representing Joints
        Jointi = joint_centers[-1].locatenew('jont' + str(i + 1), l[i] * Li.y)
        Jointi.v2pt_theory(joint_centers[-1], In_frame, Li)
        joint_centers.append(Jointi)
        #adding forces
        forces.append((Pi, -m[i + 1] * g * In_frame.y))

        #adding cential inertias
        IDi = me.inertia(frames[i + 1], 0, 0, J[i + 1])
        ICi = (IDi, mass_centers[i + 1])
        central_inertias.append(ICi)
        LBodyi = me.RigidBody('L' + str(i + 1) + '_Body', mass_centers[i + 1],
                              frames[i + 1], m[i + 1], central_inertias[i + 1])
        rigid_bodies.append(LBodyi)
        kindiffs.append(q[i + 1].diff(t) - qdot[i + 1])
    # add torques :
    if n == 1 :
        torqueVector1 = -(d[0] *qdot[1]) * In_frame.z
        torques =[(frames[1],torqueVector1)]
    elif n== 2 :
        torqueVector1 = -(d[0] *qdot[1] - d[1]*(qdot[1]-qdot[2])) * In_frame.z
        torqueVector2 = -(d[1] *(qdot[2]-qdot[1])) * In_frame.z
        torques =[(frames[1],torqueVector1), (frames[2],torqueVector2)]
    elif n== 3 :
        torqueVector1 = (-d[0] *qdot[1] - d[1]*(qdot[1]-qdot[2])) * In_frame.z
        torqueVector2 = (-d[1] *(qdot[2]-qdot[1]) - d[2]*(qdot[2]-qdot[3])) * In_frame.z
        torqueVector3 = -d[2] *(qdot[3]- qdot[2]) * In_frame.z
        torques =[(frames[1],torqueVector1), (frames[2],torqueVector2), (frames[3],torqueVector3)]

    #generalized force
    loads = torques + forces
    #Kane's Method --> Equation of motion
    Kane = me.KanesMethod(In_frame, q, qdot, kd_eqs=kindiffs)
    fr, frstar = Kane.kanes_equations(rigid_bodies, loads)
    mass_matrix = sm.trigsimp(Kane.mass_matrix_full)
    forcing_vector = sm.trigsimp(Kane.forcing_full)

        #xdot_expr=(mass_matrix.inv()*forcing_vector)

        # defining parameter values according to number of pendulum n :
    if param_deviation == True:
        param_dict_with_deviation = ct.model.param_dict_with_deviation

        param_values = [param_dict_with_deviation[str(li)] for li in l] + [
            param_dict_with_deviation[str(ai)] for ai in a
        ] + [param_dict_with_deviation[str(mi)]
             for mi in m] + [param_dict_with_deviation[str(Ji)] for Ji in J] + [
                 param_dict_with_deviation[str(di)] for di in d
             ] + [param_dict_with_deviation['g']
                  ] + [param_dict_with_deviation['f']]

    else:
        param_values = ct.parameter_values

    param_symb = list(l + a + m + J + d + (g, f))
    param_list = zip(param_symb, param_values)
    param_dict = dict(param_list)

    # substituting parameters to mass and forcing_vector
    mass_matrix_simplified = mass_matrix.subs(param_dict).simplify()
    forcing_vector_simplified = forcing_vector.subs(param_dict).simplify()

    # finding fx and gx wiht qdd0 as input
    mass_matrix44= mass_matrix_simplified[len(q):,len(q):]
    forcing_vector44= forcing_vector_simplified[len(q):,0]
    fx, gx = generate_state_equ(mass_matrix44, forcing_vector44, dynamic_symbs)

    print('system model succesfully finished !')
    # returning the model in a container :
    ct.model.t = t
    ct.model.q = q
    ct.model.qdot = qdot
    ct.model.qdd = qdd
    ct.model.dynamic_symbs = dynamic_symbs
    ct.model.f = f
    ct.model.u = u
    ct.model.m = m
    ct.model.J = J
    ct.model.l = l
    ct.model.a = a
    ct.model.d = d
    ct.model.g = g
    ct.model.param_dict = param_dict
    ct.model.frames = frames
    ct.model.mass_centers = mass_centers
    ct.model.origin_point = O
    ct.model.joint_centers = joint_centers
    ct.model.central_inertias = central_inertias
    ct.model.rigid_bodies = rigid_bodies
    ct.model.forces = forces
    ct.model.torques = torques
    ct.model.loads = loads
    ct.model.kindiffs = kindiffs
    ct.model.fr = fr
    ct.model.fstar = frstar
    ct.model.mass_matrix = mass_matrix
    ct.model.mass_matrix_simplified = mass_matrix_simplified
    ct.model.forcing_vector = forcing_vector
    ct.model.forcing_vector_simplified = forcing_vector_simplified
    ct.model.fx = fx
    ct.model.gx = gx



    # thats just a trick to be able to dump the results (there is a problem with dumping classes outside of the main!)
    for qi in q:
        qi.__class__.__module__ = '__main__'
    for qdoti in qdot:
        qdoti.__class__.__module__ = '__main__'
    for qddi in qdd:
        qddi.__class__.__module__ = '__main__'

    f.__class__.__module__ = '__main__'

    label = ct.label
    # storing system model as binary file to be used later

    param_tol_dict = ct.model.param_tol_dict if param_deviation else None
    default_tol= ct.model.default_tol if param_deviation else '0.0'

    sys_model = {
        'fx': fx,
        'gx': gx,
        'q': q,
        'qdot': qdot,
        'qdd': qdd,
        'dynamic_symbs': dynamic_symbs,
        'param_dict': param_dict,
        'param_tol_dict': param_tol_dict ,
        'default_tol': default_tol
    }


    if param_deviation == True:
        model_file_name= ct.model.model_file_name

    else:
        model_file_name= 'sys_model_' + 'without_param_deviation_'+ label + '.pkl'

    dump_model(model_file_name, sys_model)



    # ipydex.IPS()
