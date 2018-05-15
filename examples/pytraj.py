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
import pytrajectory as pytr
#import pycartan as pc
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

#endulum frame
pen_frame = In_frame.orientnew('L1', 'Axis', [q[1], In_frame.z])
pen_frame.set_ang_vel(In_frame, qdot[1] * In_frame.z)
#mass center of pendulum
s = C0.locatenew('a', a * pen_frame.x)
s.v2pt_theory(C0, In_frame, pen_frame)


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

mod.calc_state_eq(simplify=True)
xx=mod.xx
fx=mod.f
G=mod.g

equilib_point=sm.Matrix([0, sm.pi/3, 0, 0])
parameter_values = [(g, 9.81), (a, 0.2), (d, 0.0), (m[0], 3.34), (m[1], 0.8512), (J[0], 0), (J[1], 0.01980)]
#replm = list(map(lambda a,b :(a,b),xx, equilib_point)) + parameter_values
replm = parameter_values
frames_per_sec= 60
final_time= 5
tt=np.linspace(0.0, final_time, final_time*frames_per_sec)
xx0= st.to_np(equilib_point).ravel()

sim=st.SimulationModel(mod.f, mod.g , mod.xx, model_parameters=parameter_values)

def rhs(x,u):
    f_func=st.expr_to_func(xx, fx, np_wrapper=True)
    g_func=st.expr_to_func(xx, G, np_wrapper=True)
    ret=f_func+g_func.dot(np.array([0, u]))
    return ret





xa=[0.0, np.pi/2*(-1),0.0, 0.0 ]
xb=[0.0, np.pi/2, 0.0, 0.0]
ua=[0, 0]
ub=[0, 0]
control_sys=pytr.ControlSystem(rhs,a=0, b=2.0, xa=xa, xb=xb, ua=ua, ub=ub )
x, u=control_sys.solve()


