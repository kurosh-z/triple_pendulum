import numpy as np
from scipy import sin, cos, pi
from scipy.integrate import odeint
import matplotlib as plt
from pytrajectory import ControlSystem
from pytrajectory.Visualization import Animation
#nonlinear Model

'''
def x_dot(x, t, vect, uS):

    y, y_dot, phi, phi_dot = x

    u = np.interp(t, vect, uS)

    l = 1
    g = 9.8
    a1 = 3 * g / 2 * g
    a2 = 3 / 2 * l

    ff = np.array([y_dot, u, phi_dot, a1 * sin(phi) - a2 * cos(phi) * u])

    return ff


tf = 2.0
steps = 200
vect = np.linspace(0, tf, steps)
c = [-135.128, 625.696, -888.698, 497.538, -96.3861]
uS = c[0] * vect + c[1] * vect**2 + c[2] * vect**3 + c[3] * vect**4 + c[4] * vect**5

x0 = [0, 0, -pi, 0]

# Integrating to find xS
xS = odeint(x_dot, x0, vect, args=(vect, uS))


Q = np.identity(4)
R = 0.01
S = np.identity(4)


def riccati_dgl(t, P, vect, uS, xS):

    l = 1
    g = 9.8
    a1 = 3 * g / 2 * g
    a2 = 3 / 2 * l

    phiS = np.interp(t, vect, xS[:, 2])
    us = np.interp(t, vect, uS)

    A = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, a1 * cos(phiS) + a2 * sin(phiS) * us,0 ]
    ])
    b = np.array([0, 1, 0, a2 * cos(phiS)]).T

    P_dot = -P * A - A.T * P + P * b * R**-1 * b.T * A

    return P_dot

odeint(riccati_dgl, )
P0 = np.array([1, 1, 1, 1])

P = odeint(riccati_dgl, P0, vect[::-1], args=(vect[::-1], uS, xS))

#Psim=P[::-1]


# K=np.zeros((4,steps))
# for t in vect :
#     K[:,t]=-(1/R)*b[t].T*P[]
'''