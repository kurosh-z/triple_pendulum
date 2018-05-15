from __future__ import division, print_function
import sympy as sm
from sympy import trigsimp
from sympy import latex
import sympy.physics.mechanics as me
import mpmath as mp
import symbtools as st
import numpy as np
import symbtools.modeltools as mt
import symbtools.noncommutativetools as nct
from scipy.integrate import odeint

import pycartan as pc
from sympy.solvers import solve

from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

#definingymbolic variables
q = st.symb_vector('q:2')
qdot = st.time_deriv(q, q)
qdd = st.time_deriv(qdot, qdot)
st.make_global(q, qdot)

f = sm.symbols('f')
m = sm.symbols('m:2')
J = sm.symbols('J:2')
l = sm.symbols('l')
a = sm.symbols('a')
d = sm.symbols('d')
g, t = sm.symbols('g t')

# defining reference frames and points
In_frame = me.ReferenceFrame('In_frame')  #reference frame with velocity 0
O = me.Point('O')
O.set_vel(In_frame, 0)

# Mass center of the cart
C0 = me.Point('C0')
C0.set_pos(O, q[0] * In_frame.x)
C0.set_vel(In_frame, qdot[0] * In_frame.x)
#cart frame
cart_frame = In_frame.orientnew('L0', 'Axis', [q[0], In_frame.x])
cart_frame.set_ang_vel(In_frame, 0)
'''
cart_inertia_dyadic = me.inertia(cart_frame, 0, 0, J[0])
cart_central_inertia = (cart_inertia_dyadic, C0)

cart = me.RigidBody('Cart', C0, cart_frame, m[0], cart_central_inertia)
'''
#endulum frame
pen_frame = In_frame.orientnew('L1', 'Axis', [q[1], In_frame.z])
pen_frame.set_ang_vel(In_frame, qdot[1] * In_frame.z)
#mass center of pendulum
s = C0.locatenew('a', a * pen_frame.x)
s.v2pt_theory(C0, In_frame, pen_frame)
'''
pen_inertia_dyadic = me.inertia(pen_frame, 0, 0, J[1])
pen_central_inertia = (pen_inertia_dyadic, a)
pen = me.RigidBody('Pen', a, pen_frame, m[1], pen_central_inertia)
'''

# finding velocities in inertial reference frame --> use them to find kinetic energy
v0 = C0.vel(In_frame)
v1 = s.vel(In_frame)
T = 1 / 2 * m[0] * v0.dot(v0) + 1 / 2 * m[1] * v1.dot(
    v1) + 1 / 2 * J[1] * qdot[1]**2

# potential energy
h = s.pos_from(O).dot(In_frame.y)
#V = 1 / 2 * m[0] * g * h - 1/2 * d * qdot[1]**2
V = 1 / 2 * m[0] * g * h
tau = -d * qdot[1]

params = [m[0], m[1], J[0], J[1], a, d, g]
st.make_global(params)

mod = mt.generate_symbolic_model(T, V, q, [f, tau])
mod.eqns.simplify()
solutions = solve(mod.eqns, q[0], q[1], qdot[0], qdot[1])
print(solutions)