import numpy as np
from numpy import dot, arange, around
from scipy import sin, cos, pi
from scipy.integrate import odeint
from odeintw import odeintw
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle  
from matplotlib import animation
from visualisation22 import Animation

#from Visualization.visualisation import Animation

#nonlinear model


def x_dot(x, t, vect, uS, Tracking=False, xS=None, K=None):

    y, y_dot, phi, phi_dot = x
    c = [-135.128, 625.696, -888.598, 497.538, -96.3861]
    us = c[0] * t + c[1] * t**2 + c[2] * t**3 + c[3] * t**4 + c[4] * t**5

    if Tracking == False:
        #u = np.interp(t, vect, uS)
        u = us
    else:

        ys = np.interp(t, vect, xS[:, 0])
        y_dots = np.interp(t, vect, xS[:, 1])
        phis = np.interp(t, vect, xS[:, 2])
        phi_dots = np.interp(t, vect, xS[:, 3])
        xs = np.array([ys, y_dots, phis, phi_dots])
        k1 = np.interp(t, vect, K[:, 0])
        k2 = np.interp(t, vect, K[:, 1])
        k3 = np.interp(t, vect, K[:, 2])
        k4 = np.interp(t, vect, K[:, 3])
        k = np.vstack([k1, k2, k3, k4])
        u = us + dot(k.T, (xs - x))

    l = 1
    g = 9.81
    a1 = 3 * g / (2 * l)
    a2 = 3 / (2 * l)

    ff = np.array([y_dot, u, phi_dot, a1 * sin(phi) - a2 * cos(phi) * u])
    # print('u:\n',u)
    # print('us:',us)
    # print('t:',t)
    # print('x_dot',ff)
    return ff


l = 1
g = 9.81
a1 = 3 * g / 2 * l
a2 = 3 / 2 * l
tf = 2.0
steps = 200
vect = np.linspace(0, tf, steps)
c = [-135.128, 625.696, -888.598, 497.538, -96.3861]
uS = c[0] * vect + c[1] * vect**2 + c[2] * vect**3 + c[3] * vect**4 + c[4] * vect**5

x0 = [0, 0, -pi, 0]

# Integrating to find xS
xS = odeint(x_dot, x0, vect, args=(vect, uS, False))
# print('vect :',vect[::-1],end='\n')
# print('xS :', xS,end='\n')

S = np.identity(4)


def riccati_dgl(P, t, vect, uS, xS):

    l = 1
    g = 9.81
    a1 = 3 * g / 2 * l
    a2 = 3 / 2 * l

    Q = np.identity(4)
    R = 0.01

    phiS = np.interp(t, vect, xS[:, 2])
    # print('phiS:', phiS)
    us = np.interp(t, vect, uS)
    # print('us : ',us)

    A = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1],
                  [0, 0, a1 * cos(phiS) + a2 * sin(phiS) * us, 0]])

    b = np.array([0, 1, 0, -a2 * cos(phiS)]).T

    P_dot = -P * A - A.T * P + P * b * R**-1 * b.T * P - Q

    # Debugging Prints :
    # print('t= ',t)
    # print('P:\n',P)
    # print('P:\n',-P)
    # print('A: \n',A)
    # print('2nd term :\n',P * b * R**-1 * b.T * P )
    # print('P_dot \n :', P_dot)

    return P_dot


P0 = S
P = odeintw(riccati_dgl, P0, vect[::-1], args=(vect[::-1], uS, xS))
Psim = P[::-1]

Klist = []
for i in range(steps):
    phiS = xS[i, 2]
    us = uS[i]
    R = 0.01
    A = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1],
                  [0, 0, a1 * cos(phiS) + a2 * sin(phiS) * us, 0]])

    b = np.array([0, 1, 0, -a2 * cos(phiS)]).T

    k = list([dot(b.T, Psim[i])])
    Klist.append(k)

K = np.vstack(Klist[i] for i in range(steps))

from ipydex import IPS
#IPS()
x0 = [0, 0, -pi, 0]
xCl = odeint(x_dot, x0, vect, args=(vect, uS, True, xS, K))

#plotting the results
fig, axes = plt.subplots()
axes.plot(vect, xS[:, 2] * 180 / pi, 'o')
axes.plot(vect, K)

plt.show()

#axes.plot(vect,xS[:,2]-xCl[:,2],'b')
#comparing results in a matrix :
#np.vstack([xS[:, 2] * 180 / pi, xCl[:, 2] * 180 / pi]).T



def animate_pendulum(t, states, xS, filename=None):
    fig = plt.figure()

    x = states[:, 0]
    phi = states[:, 2]
    xs=xS[:,0]
    phis=xS[:,2]

    

    cart_width = 0.2
    cart_hight = 0.1
    rod_length = 0.5
    

    x_cart = x
    y_cart = 0
    
    #Trajecotry of the controled system 
    x_pendulum = rod_length * sin(phi) + x_cart
    y_pendulum = rod_length * cos(phi)
    
    #Reference curve to be tracked
    curveXes=rod_length * sin(phis) + xs
    curveYes= rod_length * cos(phis)


    #xmin = around(states[:, 0].min() - cart_width / 2, 1)
    #xmax = around(states[:, 0].max() + cart_width / 2, 1)
    xmin = -1.5
    xmax = 1.5
    #create the axes
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-.7, .7), aspect='equal')

    #Display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    #Create a rectangular cart
    rect = Rectangle(
        (x_cart[0] - 0.5 * cart_width, y_cart - .05 * cart_hight),
        cart_width,
        cart_hight,
        fill=True,
        facecolor='gray',
        linewidth=0.2)

    ax.add_patch(rect)

    #blank line for pendulum
    line, = ax.plot([], [], lw=2, marker='o', markersize=6)
    #blank desired curve 
    curve, =ax.plot([], [], lw=1,color='r', animated=True )
    xCurve, yCurve=[], []


    #initialization function : plot the background of teach frame
    def init():
        time_text.set_text('')
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        curve.set_data([],[])
        return time_text, rect, line, curve, 

    #animate fucntion: updating the objects
    def animate(i):
        time_text.set_text('time :{:2.2f}'.format(t[i]))
        rect.set_xy((x[i] - cart_width / 2.0, -cart_hight / 2))
        line.set_data((x[i],x_pendulum[i]),(0, y_pendulum[i]))
        xCurve.append(curveXes[i])
        yCurve.append(curveYes[i])
        curve.set_data(xCurve, yCurve)

        return time_text, rect, line, curve, 

    #call the animator function
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(t),
        init_func=init,
        interval=t[-1] / len(t) ,
        blit=True,
        repeat=False)
    
    
    if filename is not None:
        anim.save(filename,fps=80, extra_args=['-vcodec', 'libx264']) 

animate_pendulum(vect, xCl, xS, filename="einfachpendel.mp4")

















"""
def draw(xt, image):

    x = xt[0]
    phi = xt[2]
    cart_width = 0.05
    cart_hight = 0.02
    rod_length = 0.5
    pendulum_size = 0.15

    x_cart = x
    y_cart = 0

    x_pendulum = rod_length * sin(phi) + x_cart
    y_pendulum = rod_length * cos(phi)

    pendulum = mpl.patches.Circle(
        xy=(x_pendulum, y_pendulum), radius=pendulum_size, color='black')
    cart = Rectangle(
        (x_cart - 0.5 * cart_width, y_cart - .05 * cart_hight),
        cart_width,
        cart_hight,
        fill=True,
        facecolor='gray',
        linewidth=0.2)

    joint = Circle((x_cart, 0), 0.005, color='black')
    rod = mpl.lines.Line2D(
        [x_cart, x_pendulum], [y_cart, y_pendulum],
        color='black',
        zorder=1,
        linewidth=2.0)

    image.patches.append(pendulum)
    image.patches.append(cart)
    image.patches.append(joint)
    image.lines.append(rod)

    return image


# t=np.array(vect)
# x1=np.array(xCl[:,0])
# x2=np.array(xCl[:,1])
# x3=np.array(xCl[:,2])
# x4=np.array(xCl[:,3])
# u=np.array(uS)

sim_data = [vect, xCl[:, 0], xCl[:, 1], xCl[:, 2], xCl[:, 3], uS]

A = Animation(
    drawfnc=draw,
    simdata=sim_data,
)
#plotsys=[(0,'x'), (2,'phi')], plotinputs=[(0,'u')]
#xmin = np.min(sim_data[1][:,0]); xmax = np.max(sim_data[1][:,0])
#A.set_limits(xlim=(xmin - 0.5, xmax + 0.5), ylim=(-0.6,0.6))

A.animate()
"""