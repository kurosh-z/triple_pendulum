# coding: utf-8
'''
main function 
Developer:
-----------
Kurosh Zamani

-----------
ToDo :
  


'''
#=============================================================
# Standard Python modules
#=============================================================
import os, sys
import logging
import dill as pickel

#=============================================================
# External Python modules
#=============================================================
from sympy import sin, cos, Function
import sympy as sp
import sys
import numpy as np
import symbtools as st
from scipy.integrate import odeint
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
from pint import UnitRegistry
ureg = UnitRegistry()

import time
import ipydex

#===============================================================
# model test : comparing lagrange and Kane model to ensure they
#              are correct !
#===============================================================

def state_lagrange_matlab(xx, uu):
    q0, q1, q2, q3, q0d, q1d, q2d, q3d= xx
    u, = uu

    from numpy import cos, sin

    
    
    uu = u 

    t2 = q1*2.0 
    t3 = q2*2.0 
    t4 = q3*2.0 
    t5 = t3-t4 
    t6 = cos(t5) 
    t7 = q1+q2-t4 
    t8 = cos(t7) 
    t9 = q1+q3-t3 
    t10 = cos(t9) 
    t11 = q1d**2 
    t12 = q1-t3+t4 
    t13 = q1+t3-t4 
    t14 = q2d**2 
    t15 = q3d**2 
    t16 = q1-q2 
    t17 = cos(t16) 
    t18 = q1-q3 
    t19 = cos(t18) 
    t20 = q1-t3 
    t21 = t2-t3 
    t22 = t2-t4 
    t23 = sin(t18) 
    t24 = q2-q3 
    t25 = cos(t21) 
    t26 = t25*7.491382592641777e109 
    t27 = cos(t22) 
    t28 = t6*2.271682403688436e109 
    t33 = t27*2.315373445484391e108 
    t29 = t26+t28-t33-1.843157511455665e110 
    t30 = 1.0 /t29 
    t31 = sin(t16) 
    t32 = sin(t24) 
    t34 = sin(q1) 
    t35 = cos(q1) 
    q1dd = (q1d*1.015813267419571e109-q2d*2.768080586187006e105-t34*5.677668304247255e111+sin(q1-q3*2.0)*7.098066718813087e109+sin(t12)*3.507177517017995e110+sin(t13)*3.507177517017995e110-sin(t20)*2.296576976056745e111-q1d*t6*1.873188154316937e108+q2d*t6*5.104418233689518e104-q1d*t8*3.898362374178152e104+q2d*t8*3.295862141465522e107-q3d*t8*3.291963779091344e107-q2d*t10*1.30091021758609e108+q3d*t10*1.30091021758609e108+q1d*t17*2.321700609161756e105-q2d*t17*1.962876820338401e108+q3d*t17*1.960555119729239e108+q2d*t19*1.101659074407529e108-q3d*t19*1.101659074407529e108+t15*t23*1.948570052405966e109+t14*t31*1.78125099029818e110-t35*u*5.787633337662849e110+u*cos(t12)*3.575104502566763e109+u*cos(t13)*3.575104502566763e109+u*cos(t20)*2.341057060200555e110-t14*sin(t7)*3.031692105181125e108+t15*sin(t9)*2.482182706405859e109+t11*sin(t21)*7.491382592641777e109-t11*sin(t22)*2.315373445484391e108-u*cos(q1-t4)*7.235542017138723e108) /(cos(q2*-2.0+t2)*7.491382592641777e109-cos(q3*-2.0+t2)*2.315373445484391e108+cos(q3*-2.0+t3)*2.271682403688436e109-1.843157511455665e110) 

    t36 = cos(t24) 
    t37 = t36*6.055033876876412e-2 
    t38 = q2+q3-t2 
    t39 = cos(t38) 
    t40 = t37-t39*2.41937458625024e-2 
    t41 = q1d*1.9497e-6 
    t42 = q3d*1.64642e-3 
    t43 = sin(q2) 
    t44 = t43*4.63555950998337 
    t45 = t11*t31*1.5121091164064e-1 
    t46 = cos(q2) 
    t47 = t46*u*4.72534098877e-1 
    t48 = q2d*(-1.6483697e-3)+t41+t42+t44+t45+t47-t15*t32*5.0101981210107e-2 
    t49 = q1d*7.1548897e-3 
    t50 = q2d*1.9497e-6 
    t51 = t34*6.21950971362624 
    t52 = t14*t31*1.5121091164064e-1 
    t53 = t15*t23*3.826404292896e-2 
    t54 = t49-t50-t51+t52+t53-t35*u*6.33996912704e-1 
    t55 = q2d*1.64642e-3 
    t56 = sin(q3) 
    t57 = t56*1.17303206604093 
    t58 = t11*t23*3.826404292896e-2 
    t59 = t14*t32*5.0101981210107e-2 
    t60 = cos(q3) 
    t61 = t60*u*1.19575134153e-1 
    q2dd = t30*t40*(q3d*(-1.64642e-3)+t55+t57+t58+t59+t61)*2.494247425481559e112+t30*t54*(t8*5.99095112452963e-3-t17*3.567958424647751e-2)*3.337479743626422e112+t30*t48*(t19**2*1.464136981269294e-3-8.91787336727008e-3)*2.085924839766514e113 

    q3dd = t30*(t17**2*2.286473979919344e-2-3.697780670566373e-2)*(-t42+t55+t57+t58+t59+t61)*2.085924839766514e113+t30*t54*(t10*9.89958937147315e-2-t19*8.383339846640876e-2)*7.981591761540989e111+t30*t40*t48*2.494247425481559e112 


    
    return np.array([q0d, q1d, q2d, q3d, u, q1dd, q2dd, q3dd ])


def lagrange_model():
    '''

    '''

    t = sp.Symbol('t')
    params = sp.symbols('m_0, m_1, m_2, m_3, J_1, J_2, J_3, l_1, l_2, l_3, a_1, a_2, a_3, g, d_1, d_2, d_3')
    m0, m1, m2, m3, J1, J2, J3, l1, l2, l3, a1, a2, a3, g, d1, d2, d3 = params
    
    F = sp.Symbol('F')
    
    q1t = Function('q1')(t)
    q2t = Function('q2')(t)
    q3t = Function('q3')(t)
    q4t = Function('q4')(t)
    
    q1dt = q1t.diff(t)
    q2dt = q2t.diff(t)
    q3dt = q3t.diff(t)
    q4dt = q4t.diff(t)
    
    q1ddt = q1t.diff(t,2)
    q2ddt = q2t.diff(t,2)
    q3ddt = q3t.diff(t,2)
    q4ddt = q4t.diff(t,2)
    
    P1 = sp.Matrix([q4t + a1 * sin(q1t), +a1 * cos(q1t)])
    P2 = sp.Matrix(
        [q4t + l1 * sin(q1t) + a2 * sin(q2t), l1 * cos(q1t) + a2 * cos(q2t)])
    P3 = sp.Matrix([
        q4t + l1 * sin(q1t) + l2 * sin(q2t) + a3 * sin(q3t),
        +l1 * cos(q1t) + l2 * cos(q2t) + a3 * cos(q3t)
    ])
    
    
    P1dt = P1.diff(t)
    P2dt = P2.diff(t)
    P3dt = P3.diff(t)
    
    # Kinetische Energie vom Schlitten
    T0 = 0.5 * m0 * q4dt**2
    # Kinetische Energie vom Pendel1
    T1 = 0.5 * m1 * (P1dt.T * P1dt)[0] + 0.5 * J1 * q1dt**2
    # Kinetische Energie vom Pendel2
    T2 = 0.5 * m2 * (P2dt.T * P2dt)[0] + 0.5 * J2 * q2dt**2
    # Kinetische Energie vom Pendel3
    T3 = 0.5 * m3 * (P3dt.T * P3dt)[0] + 0.5 * J3 * q3dt**2
    
    # Gesamt Kinetische Energie
    T = T0 + T1 + T2 + T3
    
    # Gesamt Potentielle Energie
    U = g * (m1 * P1[1] + m2 * P2[1] + m3 * P3[1])
    
    # Geschwindigkeitsabhängige (viskose) Reibungen
    R = 0.5 * d1 * q1dt**2 + 0.5 * d2 * (q2dt - q1dt)**2 + 0.5 * d3 * (q3dt - q2dt)**2
    
    
    #Lagrange-Funktion
    L = T - U
    L = L.expand()
    L = sp.trigsimp(L)
    
    # --- Lagrange-Gleichungen ---
    # Hilfsterme:
    L_d_q1 = L.diff(q1t)
    L_d_q2 = L.diff(q2t)
    L_d_q3 = L.diff(q3t)
    L_d_q4 = L.diff(q4t)
    
    L_d_q1dt = L.diff(q1dt)
    L_d_q2dt = L.diff(q2dt)
    L_d_q3dt = L.diff(q3dt)
    L_d_q4dt = L.diff(q4dt)
    
    DL_d_q1dt = L_d_q1dt.diff(t)
    DL_d_q2dt = L_d_q2dt.diff(t)
    DL_d_q3dt = L_d_q3dt.diff(t)
    DL_d_q4dt = L_d_q4dt.diff(t)
    
    R_d_q1dt = R.diff(q1dt)
    R_d_q2dt = R.diff(q2dt)
    R_d_q3dt = R.diff(q3dt)
    R_d_q4dt = R.diff(q4dt)
    
    Eq1 =  DL_d_q1dt - L_d_q1 + R_d_q1dt
    Eq2 =  DL_d_q2dt - L_d_q2 + R_d_q2dt
    Eq3 =  DL_d_q3dt - L_d_q3 + R_d_q3dt
    Eq4 =  DL_d_q4dt - L_d_q4 + R_d_q4dt - F
    
    Eq = sp.Matrix([Eq1, Eq2, Eq3, Eq4]) 
    
    # Symbole für Zeitfunktionen einsetzen (für bessere Übersicht)
    # Symbole anlegen
    q_symbs = sp.symbols("q1, q2, q3, q4, q1d, q2d, q3d, q4d, q1dd, q2dd, q3dd, q4dd")
    q1, q2, q3, q4, q1d, q2d, q3d, q4d, q1dd, q2dd, q3dd, q4dd = q_symbs
    # Substitution festlegen
    # Auchtung: Reihenfolge ist wichtig:
    # Mit höchsten Ableitungen beginnen, weil sonst die Funktionen innerhalb dieser
    # Ableitungen ersetzt werden
    subs_list=[(q1ddt, q1dd), (q2ddt, q2dd), (q3ddt, q3dd), (q4ddt, q4dd), (q1dt, q1d),(q2dt, q2d),(q3dt, q3d),(q4dt, q4d), (q1t, q1),(q2t, q2),(q3t, q3),(q4t, q4)]
    #Substitution durchführen
    Eq = Eq.subs(subs_list)
    
    qq=sp.Matrix([q1, q2, q3, q4])
    # Vektor der Beschleunigungen
    acc = sp.Matrix([q1dd, q2dd, q3dd, q4dd])
    # Massenmatrix
    MM = Eq.jacobian(acc)
    acc0 =list(zip(acc, [0,0,0,0]))
    
    rest = Eq.subs(acc0) # Alles was übrig bleibt, wenn Beschleunigungen 0 sind
    rest.simplify()
    
    a = sp.Symbol('a') # neuer virtueller Eingang
    MM3x3 = MM[:-1,:-1]
    
    # Inverse der Massenmatrix bilden  (Abkürzung einführen)
    MM3x3det = MM3x3.det()
    D = sp.Symbol("D")
    M3x3inv = MM3x3.adjugate()/D
    
    phi_dd_show = -M3x3inv * (a * MM[:-1, -1] + rest[:-1,0])
    
    phi_dd = phi_dd_show.subs(D, MM3x3det)
    
    dot_state = sp.Matrix([q1d, q2d, q3d, q4d] + phi_dd[:] + [a] )
    
    params_values = [(m0, 3.34), (m1, 0.8512), (m2, 0.8973), (m3, 0.5519), 
                     (J1, 0.01980194), (J2, 0.02105375), (J3, 0.01818537), 
                     (l1, 0.32), (l2, 0.419), (l3, 0.485), 
                     (a1, 0.20001517), (a2, 0.26890449), (a3, 0.21666087), (g, 9.81),
                     (d1, 0.00715294), (d2, 1.9497e-06), (d3, 0.00164642)]
    

    phi_dd= phi_dd.subs(dict(params_values))
    
    dot_state =sp.Matrix([q1d, q2d, q3d, q4d] + phi_dd[:] + [a] )
    qdd_part_lin_num= list(dot_state)


    
    q1, q2, q3, q4, q1d, q2d, q3d, q4d = q_symbs[:-4]
    q1dd_expr, q2dd_expr, q3dd_expr, q4dd_expr = qdd_part_lin_num[-4:]

    

    q1dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q1dd_expr, 'numpy')
    q2dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q2dd_expr, 'numpy')
    q3dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q3dd_expr, 'numpy')
    q4dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q4dd_expr, 'numpy')

    def state_eq_lagrange(xx, aa):
        q4, q1, q2, q3, q4d, q1d, q2d, q3d= xx
        a, = aa
        qdd1=q1dd_fnc(q1, q2, q3, q4, q1d, q2d, q3d, q4d, a)
        qdd2=q2dd_fnc(q1, q2, q3, q4, q1d, q2d, q3d, q4d, a)
        qdd3=q3dd_fnc(q1, q2, q3, q4, q1d, q2d, q3d, q4d, a)
        
        return np.array([q4d, q1d, q2d, q3d, a , qdd1, qdd2, qdd3])

    return state_eq_lagrange

state_func1= lagrange_model()     
def save_fig(fname, fig, FigSize):
    '''
    '''
    width_cm, height_cm = (FigSize[0] * ureg.centimeter, FigSize[1] * ureg.centimeter)
    width_inch, height_inch = (width_cm.to(ureg.inch), height_cm.to(ureg.inch))
    figsize_inch = (width_inch.magnitude, height_inch.magnitude)
    fig.set_size_inches(figsize_inch)
    # plt.show()
    # ipydex.IPS()
    plt.savefig(fname, dpi=300,bbox_inches=None)
    # os.chdir(current_path)

import cfg
from cfg import Pen_Container_initializer
from sys_model import system_model_generator
from myfuncs import sympy_states_to_func

Pen_Container_initializer(3)
ct=cfg.pendata
system_model_generator(ct)
fx=ct.model.fx
gx=ct.model.gx

state_func2= sympy_states_to_func(fx, gx)
state_func3= state_lagrange_matlab

def odeFunc(x,t, xdot_func, k):
    '''
    returns ode_func for np.odeint
    '''
    deltaX=x-np.array([0, 2*np.pi/3, 2*np.pi/3, 2*np.pi/3 , 5, 10, 10, 10])
    # u= [-k.dot(deltaX)]
    u=[2]
    # u=[0]
    # ipydex.IPS()
    xdot= xdot_func(x,u)
    return xdot

dTr=np.pi/180
rTd=180/np.pi
xx0=np.array([0, np.pi/2, np.pi/2, np.pi/2,0, 40*dTr,40*dTr , 40*dTr ])
u=[0]
# k=np.array([100,200,300,400,5000,600,800,1000])
k=np.array([110,10,10,10,10,10,10,10])
tvec=np.linspace(0, 4, 240)


# xx_lag= odeint(odeFunc, xx0, tvec, args=(state_func1, k))
xx_lag= odeint(odeFunc, xx0, tvec, args=(state_func3, k))

xx_kane= odeint(odeFunc, xx0, tvec, args=(state_func2, k))

deltaX=xx_lag-xx_kane
ipydex.IPS()

import matplotlib.pyplot as plt 

fig, axes= plt.subplots(2,1)

axes[0].plot(tvec, xx_lag[:, 1]*rTd, '--', color='b', label='$q_1lag$')
axes[0].plot(tvec, xx_lag[:, 2]*rTd, '--', color='b', label='$q_2lag$')
axes[0].plot(tvec, xx_lag[:, 3]*rTd, '--', color='b', label='$q_3lag$')

axes[0].plot(tvec, xx_kane[:, 1]*rTd, '-', color='C1', label='$q_1kane$')
axes[0].plot(tvec, xx_kane[:, 2]*rTd, '-', color='C0', label='$q_2kane$')
axes[0].plot(tvec, xx_kane[:, 3]*rTd, '-', color='C2', label='$q_3kane$')

axes[1].plot(tvec, xx_lag[:, 5]*rTd, '--', color='b', label='$\dot{q_1}lag$')
axes[1].plot(tvec, xx_lag[:, 6]*rTd, '--', color='b', label='$\dot{q_2}lag$')
axes[1].plot(tvec, xx_lag[:, 7]*rTd, '--', color='b', label='$\dot{q_3}lag$')

axes[1].plot(tvec, xx_kane[:, 5]*rTd, '-', color='C1', label='$\dot{q_1}kane$')
axes[1].plot(tvec, xx_kane[:, 6]*rTd, '-', color='C0', label='$\dot{q_2}kane$')
axes[1].plot(tvec, xx_kane[:, 7]*rTd, '-', color='C2', label='$\dot{q_3}kane$')

axes[1].set_xlabel('$t$ in $s$',fontsize=11)
axes[1].set_ylabel('$\dot{q}_i$, in $\circ$/$s$,  $_{i=1,2,3}$ ',fontsize=11)
axes[1].legend(loc=2)
axes[1].grid(True)
axes[0].grid(True)
# axes[0].set_xlabel('$t$ in $s$',fontsize=11)
axes[0].set_ylabel('$q_i$, in $s$,  $_{i=1,2,3}$ ',fontsize=11)
axes[0].legend(loc=2)
plt.subplots_adjust(hspace=0.4, wspace=0.2, left=.15)

figSize=(16, 10)
save_fig('model_test.png', fig, figSize)

plt.show()
ipydex.IPS()

