# coding: utf-8
'''
main function
Developer:
-----------
Kurosh Zamani

-----------
ToDo :



'''
#from __future__ import division, print_function
#=============================================================
# Standard Python modules
#=============================================================
import os, sys
import logging

#=============================================================
# External Python modules
#=============================================================

# from sympy.physics.vector import init_vprinting, vlatex
# init_vprinting(use_latex='mathjax', pretty_print=False)
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pint import UnitRegistry
ureg = UnitRegistry()

import cfg
from cfg import Pen_Container_initializer
from myfuncs import load_sys_model
from myfuncs import load_traj_splines
from myfuncs import load_pytrajectory_results
# from myfuncs import load_tracking_results
from myfuncs import load_traj_splines
from myfuncs import read_tracking_results_from_directories
from myfuncs import generate_model_names_from_tol_dict

import dill

import ipydex

#=============================================================
# myplot  :
# =============================================================


def myplot(ct, comparasion_mode =False):
    '''
       Plots tracking results:

       to use this function make sure your files are in a folder in
       "triple_pendulum\tracking_results" , this folder should have the
       same name as the name of your trajectory like : mat50_2.1 or 19_2.1
       (matlab trajectories : "mat + nSegment + time" ,
        pytrajectory        : seed_time)

        then you have to change some variables in deviation_list,
        model_types, xt_keys according to your files.
        the name of files will help you to find that variables!

        if you set saveFig=Ture plots will be just saved in the same
        folder as your files, otherwise it shows the plots.

        * if you have lots of models with parameter deviations, given a
        param_tol_dics, you can use generate_model_names_from_tol_dict
         to generate the list automatically !


    '''

    directories = ['mat50_2.2'] # the name of the directory is the same as name of the trajectory! TODO: change this to orgenize different plotts of the same trajectory in different folders

    # load all the tracking results from directories in directories :
    tracking_dict = read_tracking_results_from_directories(directories)
    traj_dict = ct.trajectory.parallel_res
    print('trajectory resluts successfully loaded')

    frames_per_sec = 240
    final_time = 4
    tvec2 = np.linspace(0., final_time, frames_per_sec * final_time)

    # example of deviation lists:
    # deviation_list = ['None']
    # deviation_list = ['both']
    # deviation_list = ['boundry']
    deviation_list = ['parameter']



    # tol_dicts = ct.parameter_tol_dicts
    # model_types= generate_model_names_from_tol_dict(tol_dicts)

    # model_types=['original_00'] # example for original model without deviations
    model_types= ['defaultTol_0.03'] # example of model_type with defaultTol
    # model_types= ['defaultTol_-0.03'] # example of model_type with defaultTol -0.03 

    # xt_keys=['0.5-0.03'] # example of xt_keys for time: 0.5 s , and sate deviation of 5%   
    xt_keys=['00'] # for plotting the results without boudry (state) deviations

    qr_keys= ['1.01.0']  # its not important anymore in this version just leave it like it is!

    idx=0
    figs=[]
    for directory in directories:
        for deviation in deviation_list:
            for model_type in model_types:
                for xt_key in xt_keys:
                    for qr_key in qr_keys:
                        #plot the results :
                        ploting_res(figs, directory, deviation, model_type,
                                    xt_key, qr_key, tracking_dict, traj_dict,
                                    tvec2, saveFig=True)
                        idx+=3


def ploting_res(figs, directory, deviation, model_type, xt_key, qr_key, tracking_dict, traj_dict, tvec2, saveFig=False):

    # defining datas:
    tvec = tracking_dict[directory]['simulation_time']
    k_matrix = tracking_dict[directory]['k_matrix'][deviation][model_type][xt_key][qr_key]
    u_cl = tracking_dict[directory]['u_cl'][deviation][model_type][xt_key][qr_key]
    x_cl = tracking_dict[directory]['x_cl'][deviation][model_type][xt_key][qr_key]
    deltaX = tracking_dict[directory]['deltaX'][deviation][model_type][xt_key][qr_key]
    deltaU = tracking_dict[directory]['deltaU'][deviation][model_type][xt_key][qr_key]

    x_func_traj = traj_dict[directory][0]
    u_func_traj = traj_dict[directory][1]
    # figs.append(plt.figure())
    # grid = plt.GridSpec(2, 1, hspace=0.4, wspace=0.2)

    # title of the figure :
    # plt.title(
    #     u'Trajtype: {} , dev: {} , model_type: {} , boundry_dev: {}'.
    #     format(directory, deviation, model_type, xt_key))

    # defining the postions of axex in a figure
    # ax_phis = figs[-1].add_subplot(grid[0:1, ::])
    # ax_phi_dot = figs[-1].add_subplot(grid[1::, ::])

    fig1, (ax_phis, ax_phi_dot) = plt.subplots(2, 1)
    figs.append(fig1)
    # adding subplots
    _add_subplot_to_axes( ax_phis ,"phi",tvec=tvec, tvec2=tvec2, x_cl=x_cl, x_func_traj= x_func_traj )
    _add_subplot_to_axes( ax_phi_dot ,"phi_dot",tvec=tvec, tvec2=tvec2, x_cl=x_cl, x_func_traj= x_func_traj )
    plt.subplots_adjust(hspace=0.4, wspace=0.2, left=.15)
    # title of the figure 1 :
    fig1.suptitle(
        u'<{}><{}><{}><{}>'.format(directory, deviation, devToPrec(model_type),
                                   xtToPerc(xt_key)),
        fontsize=11)
    if saveFig:
        fname = 'phis_' + directory + '_' + deviation + '_' + model_type + '_' + xt_key + '_' + qr_key+'.png'
        figSize=(16, 10)
        save_fig(fname, figs[-1], figSize, directory)



    # plot x xdot and u :
    fig2, (ax_x, ax_x_dot, ax_u, ax_deltaU) = plt.subplots(4, 1)
    figs.append(fig2)
    fig2.suptitle(
        u'<{}><{}><{}><{}>'.format(directory, deviation, devToPrec(model_type),
                                             xtToPerc(xt_key)),
        fontsize=11)
    _add_subplot_to_axes( ax_x ,"x0",tvec=tvec, tvec2=tvec2, x_cl=x_cl, x_func_traj= x_func_traj ,label=model_type)
    _add_subplot_to_axes( ax_x_dot ,"x0_dot",tvec=tvec, tvec2=tvec2, x_cl=x_cl, x_func_traj= x_func_traj ,label=model_type)
    _add_subplot_to_axes( ax_u ,"u",tvec=tvec, u_cl=u_cl,tvec2=tvec2, u_func_traj=u_func_traj,label=model_type)
    _add_subplot_to_axes(ax_deltaU, 'deltaU', tvec=tvec, deltaU=deltaU, label=model_type)

    figs[-1].subplots_adjust(hspace=.3, left=.15)
    if saveFig:
        fname = 'cart_' + directory + '_' + deviation + '_' + model_type + '_' + xt_key + '_' + qr_key+'.png'
        figSize=(17, 16)
        save_fig(fname,figs[-1], figSize, directory)

    # plot k
    figs.append(plt.figure())
    figs[-1].suptitle(
        u'<{}><{}><{}><{}>'.format(directory, deviation, devToPrec(model_type),
                                             xtToPerc(xt_key)),
        fontsize=11)
    grid = plt.GridSpec(1, 1, hspace=0.4, wspace=0.2)
    ax_k = figs[-1].add_subplot(grid[::, ::])
    _add_subplot_to_axes( ax_k ,"k_matrix",tvec=tvec, k_matrix=k_matrix )
    if saveFig:
        fname = 'gain_' + directory + '_' + deviation + '_' + model_type + '_' + xt_key + '_' + qr_key+'.png'
        figSize=(16,7)
        save_fig(fname,figs[-1], figSize, directory)


    if not saveFig:
        plt.show()


def _add_subplot_to_axes(axes,
                         plot_name,
                         tvec,
                         tvec2=None,
                         x_cl=None,
                         u_cl=None,
                         deltaU=None,
                         deltaX=None,
                         x_func_traj=None,
                         u_func_traj=None,
                         k_matrix=None,
                         label=None):
    '''given an axes it adds charts you need
    plot_name : could be one of the folowing names :

        'phi'
        'phi_dot'
        'x0'
        'x0_dot'
        'k_matrix'
        'u'
        'deltaX'
        'deltaU'
        '''
    rTd=180/np.pi

    if x_func_traj :
        x_traj = np.array([x_func_traj(t) for t in tvec2])
    # if isinstance(u_func_traj, types.Fun :

    if plot_name == "phi" :

        phis = x_cl[:, 1:4] * rTd


        # ploting phis
        axes.plot(tvec, phis[:, 0], '-', color='C1', label='$q_1$')
        axes.plot(tvec, phis[:, 1], '-', color='C0', label='$q_2$')
        axes.plot(tvec, phis[:, 2], '-', color='C2', label='$q_3$')

        # ploting trajectories
        axes.plot(tvec2,x_traj[:, 1] * rTd, color='C1',linestyle='--',label='$q_{1ref}$')
        axes.plot(tvec2,x_traj[:, 2] * rTd, color='C0',linestyle='--',label='$q_{2ref}$')
        axes.plot(tvec2,x_traj[:, 3] * rTd, color='C2',linestyle='--',label='$q_{3ref}$')

        # add x and y labels
        axes.set_xlabel('$t$ in $s$',fontsize=11)
        axes.set_ylabel('$q_i$,  in $\circ$,  $_{i=1,2,3}$',fontsize=11)
        axes.legend(loc=1)
        axes.grid(True)

    if plot_name == "phi_dot" :

        phi_dots = x_cl[:, 5:]*rTd


        # ploting phi_dots :
        axes.plot(tvec, phi_dots[:, 0], '-', color='C1', label='$\dot{q_1}$')
        axes.plot(tvec, phi_dots[:, 1], '-', color='C0', label='$\dot{q_2}$')
        axes.plot(tvec, phi_dots[:, 2], '-', color='C2', label='$\dot{q_3}$')

        axes.plot(tvec2, x_traj[:, 5] * rTd, color='C1', linestyle='--', label='$\dot{q}_{1ref}$')
        axes.plot(tvec2, x_traj[:, 6] * rTd, color='C0', linestyle='--', label='$\dot{q}_{2ref}$')
        axes.plot(tvec2, x_traj[:, 7] * rTd, color='C2', linestyle='--', label='$\dot{q}_{3ref}$')

        axes.set_xlabel('$t$ in $s$',fontsize=11)
        axes.set_ylabel('$\dot{q}_i$, in $\circ$/$s$,  $_{i=1,2,3}$ ',fontsize=11)
        axes.legend(loc=4)

        # try :
        #     yticks= calculate_yTicks(phi_dots, yDimention=3)
        # except:
        #     pass
        # try :
        #     axes.set_yticks(yticks)
        # except:
        #     pass

        xticks= np.arange(0, 5, step=.5)
        axes.set_xticks(xticks)

        axes.grid(True)


    if plot_name == "x0" :

        x = x_cl[:, 0]

        # ploting x :
        axes.plot(tvec, x, '-',color='blue', label='$q_0$')
        axes.plot(tvec2, x_traj[:, 0], '--', label='$q_{0ref}$')

        axes.set_xlabel('$t$ in $s$', fontsize=11)
        axes.set_ylabel('$q_0$, in $m$', fontsize=11)
        axes.legend(loc=1)

        try:
            yticks= calculate_yTicks(x, label=label)
        except:
            pass
        try:
            axes.set_yticks(yticks)
        except:
            pass
        # ipydex.IPS()
        xticks= np.arange(0, 5, step=.5)
        axes.set_xticks(xticks)

        axes.grid(True)

    if plot_name == "x0_dot" :

        x_dot = x_cl[:, 4]
        try:
            yticks=calculate_yTicks(x_dot, label=label)
        except:
            pass
        xticks= np.arange(0, 5, step=.5)

        # ploting x_dot :
        axes.plot(tvec, x_dot, '-', color='blue', label='$\dot{q}_0$')
        axes.plot(tvec2, x_traj[:, 4],'--', label='$\dot{q}_{0ref}$')

        try:
            axes.set_yticks(yticks)
        except:
            pass
        axes.set_xticks(xticks)
        plt.xlim(0,4)
        axes.set_xlabel('$t$ in $s$',fontsize=11)
        axes.set_ylabel('$\dot{q}_0$,  in $m/s$',fontsize=11)
        axes.legend(loc=1)
        axes.grid(True)

    if plot_name == "u" :

        # # ploting u :

        try:
            yticks=calculate_yTicks(u_cl, label=label)
        except:
            pass

        xticks= np.arange(0, 5, step=.5)

        u_traj= np.array([u_func_traj(t) for t in tvec2])
        axes.plot(tvec, u_cl, '-', color='blue', label='$u_cl$')
        axes.plot(tvec2, u_traj,'--', label='$u_{ref}$')

        axes.set_xlabel('$t$ in $s$', fontsize=11)
        axes.set_ylabel('$u$ in $m/s^2$', fontsize=11)

        try:
            axes.set_yticks(yticks)
        except:
            pass
        if yticks == 'Warning':
            axes.set_ylim((-1e3, 1e3))
        axes.set_xticks(xticks)
        axes.set_xlim(0,4)
        axes.legend(loc=1)
        axes.grid(True)

    if plot_name == 'deltaU':

        try:
            yticks= calculate_yTicks(deltaU, label=label)
        except:
            pass

        xticks= np.arange(0, 5, step=.5)

        axes.plot(tvec, deltaU,'-',color='blue', label='$\Delta u$')
        axes.set_xlabel('$t$ in $s$', fontsize=11)
        axes.set_ylabel('$\Delta u$ in $m/s^2$', fontsize=11)

        try:
            axes.set_yticks(yticks)
        except:
            pass
        if yticks == 'Warning':
            axes.set_ylim((-1e3, 1e3))

        axes.set_xticks(xticks)
        axes.set_xlim(0,4)
        axes.legend(loc=1)
        axes.grid(True)

    if plot_name == "k_matrix" :

        # ploting K_matrix
        axes.plot(tvec, k_matrix[:, 0], '--r', color='magenta', label='$k_0$')
        axes.plot(tvec, k_matrix[:, 1], '--r', color='blue', label='$k_1$')
        axes.plot(tvec, k_matrix[:, 2], '--r', color='cyan', label='$k_2$')
        axes.plot(tvec, k_matrix[:, 3], '--r', color='black', label='$k_3$')

        axes.plot(tvec, k_matrix[:, 4], '--r', color='green', label='$k_4$')
        axes.plot(tvec, k_matrix[:, 5], '--r', color='yellow', label='$k_5$')
        axes.plot(tvec, k_matrix[:, 6], '--r', color='red', label='$k_6$')
        axes.plot(tvec, k_matrix[:, 7], '--r', color='brown', label='$k_7$')

        axes.set_xlabel('$t$ in $s$',fontsize=11)
        axes.set_ylabel('Folgereglerversarkungen')
        axes.legend(loc=4)
        axes.grid(True)


def save_fig(fname, fig, FigSize,  directory):
    '''
    '''
    current_path = os.getcwd()
    current_dir = current_path.split('/')[-1]
    if current_dir == 'core':
        os.chdir('..')
        os.chdir('tracking_results')
        os.chdir(directory)

    elif current_dir == 'triple_pendulum':
        os.chdir('tracking_results')
        os.chdir(directory)
    width_cm, height_cm = (FigSize[0] * ureg.centimeter, FigSize[1] * ureg.centimeter)
    width_inch, height_inch = (width_cm.to(ureg.inch), height_cm.to(ureg.inch))
    figsize_inch = (width_inch.magnitude, height_inch.magnitude)
    fig.set_size_inches(figsize_inch)
    # plt.show()
    # ipydex.IPS()
    plt.savefig(fname, dpi=300,bbox_inches=None)
    os.chdir(current_path)


def calculate_yTicks(y, yDimention= None, label=None):

    if yDimention :
        yMax= np.amax(y[:,0])
        yMin= np.amin (y[:,0])
        for i in range(1, yDimention):
            yMaxi= np.amax(y[:,i])
            yMini= np.amin(y[:,i])
            if yMaxi > yMax :
                yMax= yMaxi
            if yMini < yMin:
                yMin = yMini


    yMax= np.amax(y)
    yMin= np.amin(y)

    # if yMax <= 1 :
    #     stepPl= yMax/2.
    #     tmp1= np.arange(0., yMax-stepPl, step=stepPl)
    # else:
    #     stepPl= np.around(yMax/2.)
    #     tmp1= np.arange(0., np.around(yMax)-stepPl+1, step=stepPl)

    # if yMin* (-1) <= 1 :
    #     stepMi= yMin/2.
    #     tmp2= np.arange(0.,yMin-stepMi, step=stepMi)
    # else:
    #     stepMi= np.around(yMin/2.)
    #     tmp2= np.arange(0.,np.around(yMin)-stepMi, step=stepMi)

    # tmp1=np.append(tmp1, yMax)
    # tmp2=np.append(tmp2, yMin)
    # yticks= np.append(tmp2[::-1], tmp1[1::])
    # print("-------")
    # print(label)
    dist= (yMax-yMin)/4
    if np.isnan(yMin) or np.isnan(yMin):
        ret= []
        # print('nan found!'), print('-->'),print((yMin, yMax))
    elif (yMax-yMin) >= 1e6:
        ret='Warning'
        # print('got it!')
    elif (yMax)<dist:
        if (yMin< -dist):
            ret= np.array([yMin, 0])
    elif (yMin> -dist):
        if (yMax)> .4:
            ret=[0, yMax]
    elif (yMax < dist and yMin> -dist):
        ret=[]
    else:
        ret= np.array([yMin, 0, yMax])
        # print('everthing is Okay'), print('-->'), print(ret)

    # if label:
    #     ipydex.IPS()
    # ipydex.IPS()

    return ret


def devToPrec(x):
    '''
    '''
    if x== 'None':
        return 'None'
    else:
        temp=x.split('_')
        return temp[0]+"_"+ str(np.around(float(temp[1])*100, decimals=1))+" %"


def xtToPerc(tx):
    '''
    '''
    temp=tx.split(".")
    if len(temp)== 1:
        return tx
    else:
        t= temp[0]+"."+temp[1][0]+" s, "
        xx=float(temp[1][1]+"."+temp[2])*100
        xx=np.around(xx, decimals=1)
        return t+str(xx)+" %"

