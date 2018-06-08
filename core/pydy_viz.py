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
from pydy.viz.shapes import Cube, Circle, Sphere, Cylinder
import pydy.viz
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene
import sympy as sm
import sympy.physics.mechanics as me

import cfg
import ipydex

#=============================================================
# visualization  :
#=============================================================


def visualization(ct, filename=None):
    '''
    animating the resulsts 

    '''
    num_pen = ct.number_of_pendulums
    states = ct.tracking.x_closed_loop
    param_dict = ct.model.param_dict
    dynamic_symbs = ct.model.dynamic_symbs
    joint_centers = ct.model.joint_centers
    l = ct.model.l
    frames = ct.model.frames
    In_frame = frames[0]
    mass_centers = ct.model.mass_centers
    origin = ct.model.origin_point

    # defining cart's shape and frame
    cart_shape = Cube(color='red', length=0.1)
    cart_viz_frame = VisualizationFrame(In_frame, mass_centers[0], cart_shape)

    links_centers = []
    links_shapes = []
    links_viz_frames = []
    joint_shapes = []
    joint_viz_frames = []

    # definig geometric centers and frames for links
    for i in range(num_pen):
        #defining geometric centers
        LCi = me.Point('LC' + str(i))
        LCi.set_pos(joint_centers[i], l[i] / 2 * frames[i + 1].x)
        links_centers.append(LCi)

        # reference frames for links
        Bi = me.ReferenceFrame('B' + str(i))
        Bi.orient(frames[i + 1], 'Axis', [np.pi / 2, In_frame.z])

        # definging shapes for joints
        joint_shapei = Sphere(color='black', radius=0.05)
        joint_shapes.append(joint_shapei)

        # definging shapes for links
        links_shapei = Cylinder(
            radius=0.08, length=param_dict[l[i]], color='blue')
        links_shapes.append(links_shapei)

        # joint visualization frames
        joint_viz_framei = VisualizationFrame(In_frame, joint_centers[i],
                                              joint_shapei)
        joint_viz_frames.append(joint_viz_framei)

        # links visualization frames
        link_viz_framei = VisualizationFrame('link' + str(i), Bi, LCi,
                                             links_shapei)
        links_viz_frames.append(link_viz_framei)

    # defining Scenes
    scene = Scene(In_frame, origin)
    ipydex.IPS()

    scene.visualization_frames = [cart_viz_frame
                                  ] + joint_viz_frames + links_viz_frames
    scene.states_symbols = dynamic_symbs[0:4]
    scene.constants = param_dict
    scene.states_trajectories = states

    scene.display()

    # ipydex.IPS()
