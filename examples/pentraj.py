#coding: utf-8

from __future__ import division, print_function
import sympy as sm
from sympy import trigsimp
from sympy import latex
import sympy.physics.mechanics as me
import mpmath as mp
import symbtools as st
import numpy as np
from sympy import cos , sin , tan
import symbtools.modeltools as mt
import symbtools.noncommutativetools as nct
from scipy.integrate import odeint
import pytrajectory as pytr
#import pycartan as pc
from sympy.solvers import solve

from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

import ipydex


def cart_pen(x, u):
    q1, q2, q3, q4 = x
    f, = u
    
    J = 0.01980
    m0 = 3.34
    m1 = 0.8512
    l = 0.2
    g = 9.81

    alpha = (-m1 * l /
             (m0 + m1)) * cos(q1) + (J + m1 * l**2) / (m1 * l * cos(q1))

    beta = (-m1 * l / (m0 + m1) * sin(q1)) * q4**2 + g * tan(q1)

    gamma = (J + m1 * l**2) / (m1 * l * cos(q1))

    fx = np.array([[q3],[q4],[g * tan(q1) - gamma * beta / alpha], [beta / alpha]])
    gx= np.array([[0.0],[0.0], [gamma*(1/(m0+m1))* 1/alpha *f], [-1/(m0+m1)* (1/alpha)*f]])

    xdot=fx+gx

    return xdot 



#  boundary values at the start (a = 0.0 [s])
xa = [  0.0,
        -np.pi,
        0.0,
        0.0]

# boundary values at the end (b = 2.0 [s])
xb = [  0.0,
        np.pi,
        0.0,
        0.0]

# ipydex.activate_ips_on_exception()

# ipydex.IPS()      

# create trajectory object
S = pytr.ControlSystem(cart_pen, a=0.0, b=2.0, xa=xa, xb=xb)

# change method parameter to increase performance
#S.set_param('use_chains', False)

# run iteration
S.solve()