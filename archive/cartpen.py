from __future__ import division, print_function
import sympy as sm
from sympy import trigsimp
from sympy import latex
import sympy.physics.mechanics as me
import mpmath as mp


from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

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

force1 = (C0, f * In_frame.x)
force2 = 0
force3=(C0, -d*qdot[1]*In_frame.z)
force = [force1, force2]

v0=C0.vel(In_frame)
v1=a.vel(In_frame)
T = 1 / 2 * m[0] *v0.dot(v0) + 1 / 2 * m[1] *v1.dot(v1) + 1 / 2 * J[1] * qdot[1]**2

h = a.pos_from(O).dot(In_frame.y)
V = 1 / 2 * m[0] * g * h

L = T - V

q[0] = sm.Function('q[0]')
q[1] = sm.Function('q[1]')

dL_dqdot0 = sm.diff(L, qdot[0])
dL_dqdot1 = sm.diff(L, qdot[1])
dt_dL_qdot0 = sm.diff(dL_dqdot0, t)
dt_dL_dqdot1 = sm.diff(dL_dqdot1, t)

dL_dq0 = sm.diff(L, q[0])
dL_dq1 = sm.diff(L, q[1])

#lag_eq = sm.Matrix([[dt_dL_qdot0 - dL_dq0 - f], [dt_dL_dqdot1 - dL_dq1]])

E1= dt_dL_qdot0 - dL_dq0 - f
E2= dt_dL_dqdot1 - dL_dq1

qddot = me.dynamicsymbols('q:2', 2)

sols=sm.solve([E1, E2], qddot[0], qddot[1])
print ("qddot[0]= ",  (sols[qddot[0]].factor()))
