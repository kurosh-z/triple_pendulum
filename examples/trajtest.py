# acrobot

# import trajectory class and necessary dependencies
from pytrajectory import ControlSystem
import numpy as np
from sympy import cos, sin

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4 = x
    u1, = u
    
    m = 1.0             # masses of the rods [m1 = m2 = m]
    l = 0.5             # lengths of the rods [l1 = l2 = l]
    
    I = 1/3.0*m*l**2    # moments of inertia [I1 = I2 = I]
    g = 9.81            # gravitational acceleration
    
    lc = l/2.0
    
    d11 = m*lc**2+m*(l**2+lc**2+2*l*lc*cos(x1))+2*I
    h1 = -m*l*lc*sin(x1)*(x2*(x2+2*x4))
    d12 = m*(lc**2+l*lc*cos(x1))+I
    phi1 = (m*lc+m*l)*g*cos(x3)+m*lc*g*cos(x1+x3)

    ff = np.array([     x2,
                        u1,
                        x4,
                -1/d11*(h1+phi1+d12*u1)
                ])
    
    return ff


# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [  0.0,
        0.0,
        3/2.0*np.pi,
        0.0]

xb = [  0.0,
        0.0,
        1/2.0*np.pi,
        0.0]

# boundary values for the inputs
ua = [0.0]
ub = [0.0]

# create System
first_guess = {'seed' : 1529} # choose a seed which leads to quick convergence
S = ControlSystem(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=ua, ub=ub, use_chains=True, first_guess=first_guess)

# alter some method parameters to increase performance
S.set_param('su', 10)

# run iteration
S.solve()