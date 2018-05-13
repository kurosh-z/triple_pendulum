# coding: utf-8


from __future__ import division, print_function
import sympy as sm
from sympy import trigsimp
import sympy.physics.mechanics as me
import mpmath as mp

#from sympy.physics.vector import init_vprinting, vlatex
#init_vprinting(use_latex='mathjax', pretty_print=False)

q = me.dynamicsymbols('q:2')
qdot = me.dynamicsymbols('q:2', 1)
f = me.dynamicsymbols('f')
m = sm.symbols('m:2')
J = sm.symbols('J:2')
l = sm.symbols('l')
a = sm.symbols('a')
d = sm.symbols('d')
g, t = sm.symbols('g t')

In_frame = me.ReferenceFrame('In_frame')
O = me.Point('O')
O.set_vel(In_frame, 0)

C0 = me.Point('C0')
C0.set_pos(O, q[0] * In_frame.x)
C0.set_vel(In_frame, qdot[0] * In_frame.x)

cart_frame = In_frame.orientnew('L0', 'Axis', [q[0], In_frame.x])
cart_frame.set_ang_vel(In_frame, 0)

cart_inertia_dyadic = me.inertia(cart_frame, 0, 0, J[0])
cart_central_inertia = (cart_inertia_dyadic, C0)

cart = me.RigidBody('Cart', C0, cart_frame, m[0], cart_central_inertia)

pen_frame = In_frame.orientnew('L1', 'Axis', [q[1], In_frame.z])
pen_frame.set_ang_vel(In_frame, qdot[1] * In_frame.z)

a = C0.locatenew('a', a * pen_frame.x)
a.v2pt_theory(C0, In_frame, pen_frame)

pen_inertia_dyadic = me.inertia(pen_frame, 0, 0, J[1])
pen_central_inertia = (pen_inertia_dyadic, a)
pen = me.RigidBody('Pen', a, pen_frame, m[1], pen_central_inertia)

force1 = (C0, f * In_frame.x - m[0] * g * In_frame.y)
force2 = (a, -m[1] * g * In_frame.y)
#force3=(C0, -d*qdot[1]*In_frame.z)
force = [force1, force2]

cart_potential = 0
cart.potential_energy = cart_potential
#h = a.pos_from(O).dot(In_frame.y)
#pen_potential=m[1]*g*h
pen_potential = 0
cart.potential_energy = cart_potential

pen.potential_energy = pen_potential
lag = me.Lagrangian(In_frame, cart, pen)
q[0] = sm.Function('q[0]')
q[1] = sm.Function('q[1]')

lag_method = me.LagrangesMethod(
    lag, q, forcelist=force, bodies=[cart, pen], frame=In_frame)

lag_method.form_lagranges_equations()

lag_method.rhs()
