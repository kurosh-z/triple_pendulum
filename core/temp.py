import os, sys
import logging

import dill
import cfg
from cfg import Pen_Container_initializer
from myfuncs import*
import ipydex

ipydex.activate_ips_on_exception()


Pen_Container_initializer(3)
ct=cfg.pendata
label = ct.label

pfname = 'swingup_splines_' + label + '.pcl'
# load_traj_splines(cfg.pendata, pfname)
# load_pytrajectory_results(ct, pfname)
# traj_results= ct.trajectory.pytrajectory_res.items()

# types='swingup_splines'
# const={'x0':0.1, 'x4':None}
# traj_labels=[traj_results[0][0], traj_results[1][0], traj_results[2][0]]
# number_of_splines_list=[160, 320, 160]
ipydex.IPS()
split_trajectory_results(ct,pfname)
ipydex.IPS()

# dump_trajectory_results(types, traj_labels, number_of_splines_list, const, traj_results)


# pfname2='swingup_splines__add_infos__390_2_splines_320_x0_None_x4_None_.pcl'
# load_pytrajectory_results(ct, pfname2)


# pfname3 = 'swingup_splines__add_infos__29_2_splines_160_x0_None_x4_None_.pcl'
# load_pytrajectory_results(ct,pfname3)

# pfname4 = 'swingup_splines__add_infos__19_2_splines_160_x0_None_x4_None_.pcl'
# load_pytrajectory_results(ct,pfname4)
