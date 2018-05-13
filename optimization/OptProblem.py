#!/usr/bin/env python
'''
Optimizatio Problem Defition code

Developer:
-----------
Kurosh Zamani

-----------
To Do :
  -
  -
  -


'''
#=============================================================
# Standard Python modules
#=============================================================
import os, sys

#=============================================================
# External Python modules
#=============================================================
import numpy


#=============================================================
# Optimization Problem Class
#=============================================================
class Optproblem(object):
    '''
    Definition of Optimization Problem Class 

    Last update = 15,04,2018
    '''

    def __init__(self, name, obj_func, *args, **kwargs):
        '''
        Calss Initialization 


        1) Argumetns :
        - name -> STR: Name of the Problem
        - obj_func -> Func : objective Funciton

        2) Kyword argumetns:
        TO DO

        '''
        #
        self.name = name
        self.obj_func = obj_func

    # get Objective Function Method
    def getObjFunc(self):
        return self.obj_func
