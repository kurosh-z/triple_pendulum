'''
main function 
Developer:
-----------
Kurosh Zamani

-----------
To Do :
  


'''
#from __future__ import division, print_function
#=============================================================
# Standard Python modules
#=============================================================
import os
import sys
import logging
#=============================================================
# External Python modules
#=============================================================
import numpy as np
from numpy import dot, arange, around
from scipy import sin, cos, pi
import sympy as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation

import cfg
from functions import *
import ipydex

#=============================================================
# visualization  :
#=============================================================


def pen_animation(ct, filename=None):
    '''
    animating the resulsts 

    '''
    num_pen = ct.number_of_pendulums
    states = ct.tracking.x_closed_loop
    # param_values = ct.parameter_values
    param_dict = ct.model.param_dict
    l = ct.model.l
    ref_traj = ct.trajectory.cs_ret[0]

    origin = ct.model.origin_point
    frames = ct.model.frames
    In_frame = frames[0]
    joint_centers = ct.model.joint_centers
    dynamic_symbs = ct.model.dynamic_symbs
    dynamic_variables = dynamic_symbs[0:len(dynamic_symbs) - 1]
    tvec = ct.tracking.tvec

    # converting points to numpy functions of qs and qdots
    joint_centers_pos_funcs = [
        generate_position_vector_func(In_frame, origin, point,
                                      dynamic_variables)
        for point in joint_centers
    ]

    # refrence curve
    ref_curve_func = generate_position_vector_func(
        In_frame, origin, joint_centers[-1], dynamic_variables)

    ref_curve_x = np.array([ref_curve_func[0](*ref_traj(t)) for t in tvec])
    ref_curve_y = np.array([ref_curve_func[1](*ref_traj(t)) for t in tvec])

    cart_width = 0.3
    cart_hight = 0.1

    xmin = np.around(states[:, 0].min() - cart_width / 2, 1)
    xmax = np.around(states[:, 0].max() + cart_width / 2, 1)

    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.2, 1.2), aspect='equal')

    #Display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    #Create a rectangular cart
    x_cart_func=joint_centers_pos_funcs[0][0]
    y_cart_func=joint_centers_pos_funcs[0][1]
    rect = Rectangle(
        (x_cart_func(*states[0]) - 0.5 * cart_width,
         y_cart_func(*states[0]) - .05 * cart_hight),
        cart_width,
        cart_hight,
        fill=True,
        facecolor='gray',
        linewidth=0.2)

    ax.add_patch(rect)

    #blank desired curve
    curve, = ax.plot([], [], lw=1, color='r', animated=True)
    xCurve, yCurve = [], []

    #blank lines for pendulums
    
    line, = ax.plot([], [], lw=2, marker='o', markersize=6)
    #initialization function : plot the background of teach frame
    def init():
        time_text.set_text('')
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        curve.set_data([], [])
    return time_text, rect, line, curve,
    #animate fucntion: updating the objects
    def animate(i):
        
        time_text.set_text('time :{:2.2f}'.format(tvec[i]))
        rect.set_xy((joint_centers_pos_funcs[0][0](*states[i]) -
                     0.5 * cart_width,
                     joint_centers_pos_funcs[0][1](*states[i]) -
                     0.5 * cart_hight))
        x_joint1=joint_centers_pos_funcs[1][0]
        y_joint1=joint_centers_pos_funcs[1][1]
        line_x_tupel= (states[i, 0], x_joint1(*states[i]))
        line_y_tupel= (0, y_joint1(*states[i]))
        line.set_data(line_x_tupel, line_y_tupel)
        xCurve.append(ref_curve_x[i])
        yCurve.append(ref_curve_y[i])
        curve.set_data(xCurve, yCurve)
        return time_text, rect, line, curve,

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
