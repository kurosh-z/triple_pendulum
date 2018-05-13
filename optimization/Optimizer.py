#!/usr/bin/env python

'''
Optimizater code

Developer:
-----------
Kurosh Zamani

-----------
To Do :
  - pep8 beachten
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
import numpy as np
from numpy import dot
from numpy import infty
from numpy.linalg import norm
import OptProblem


#=============================================================
# Optimization Optimizer Class
#=============================================================
class Optimizer(object):
    '''
    Solver Class initialization method

    Last update = 15,04,2018
    '''

    def __init__(self, opt_problem, name={}, slv_options={}, **kwargs):
        '''
        1) Argumetns :
           - name -> STR: Name of the optimizer  *Defualt*={}
           - opt_problem -> INST : instance of Optimization problem class  
           - slv_options -> DICT : Deafult options, *Defualt*={}


        2) Keyword argumetns:
           TO DO:
            -writing slv_options

ddsfsadfds zuq
        '''
        self.name = name
        self.options = {}

        # Initializing Options
        '''
        To Do:
        -writing a options_set method to check the options
        /or set defualt values 

        -Method to check the type of opt_problem
        '''
        value_option = slv_options.keys()
        for key in value_option:
            self.options[key] = value_option[key]

        # Initializing opt_problem
        if isinstance(opt_problem, Optproblem):
            self.opt_problem = opt_problem
        else:
            raise ValueError(
                'opt_problem is not a Valid instance of Optimization Problem \n '
            )
        #end if

    # Norm Mehtod
    def _norm_infty(self, x):
        if len(x) > 0:
            return norm(x, ord=infty)
        return 0.0

    # Bactracking Armijo line Search Method
    def backtracking_armijo(self,
                            grad,
                            x,
                            s,
                            c1=1.0e-4,
                            rho=0.1,
                            alpha0=1,
                            max_iter=1000,
                            **kwargs):
        '''
        Backtracking Armijo line Search Method :

        Given a descent direction s for objective func at the current iterate x,
        compute a steplenght alpha such that
                 func(x+alpha*s)=< func(x)+c1*alpha*func_grad(x)*s
        
        To Do :
        -adding code to be able to use **kwargs variables  
        
        arguments:
        --------------

        grad: np.array of shape (n,)
              gradient of a function at point x   
        x :  np.array of shape (n,)
            coordinates of point p 
        c1: Float in range of (0,0.5),**Defualt c1=10^-4
            Armijo constant c1
        rho :Float in range of (0,1) ,**Defualt c1=0.1
            Scaling factor to reduce aplpha
        s : np.array of shape (n,) 
            A descent direction at point x (normally gradient descent, newton method or conjugate gradient)
         
        alpha0 : Float, **Defualt alpha0=1
            Initial value for alpha 
        max_iter : INT , **Defualt=1000
            Maximal number of Iterations    
            

        Outputs:
        ---------------
        alpha : Float
            Step value satisfying Armijo condition
        iter : INT
            Number of Iterations    
        

        '''
        #Check if the slope is descent direction
        if dot(grads.T, s) >= 0.0:
            raise ValueError('Direction must be a descent direction')
            return None

        #Defining new variables
        alpha = alpha0
        func = self.opt_problem.getObjFunc()
        f = func(x)
        f_plus = func(x + alpha * s)
        iter = 0
        #Loop to find a alpha
        while not (abs(f_plus) <= abs(f + c1 * alpha * dot(grad, s.T))) and (
                iter <= max_iter):
            alpha *= rho
            iter += 1
            f_plus = func(x + alpha * s)
            return alpha, iter
        #end while

    #Strong Wofe Condition Method
    def backtracking_wolfe(self,
                           x,
                           s,
                           c1=10**-4,
                           c2=0.9,
                           rho=0.1,
                           alpha0=1,
                           max_iter=1000,
                           grad_func=None,
                           **kwargs):
        '''
        Checks Armijo and Wolfe Condition(or Curvature Conditiotn)

        Wolfe Condition :
                  grad_func(x+alpha*s).T * s <= c2* grad_func(x).T *s
         or 
                          (phi_prime(alpha)) <= phi_prime(0)

        Arguments :
        ------------
        x : np.array 
           point to check Wolfe conditon 

        s: np.array 
           Descent direction at point x  (Normally Gradient Descent,Newton, Quasi_Newton or Conjugate Gradient )
        
        c1 : Float
          Armijo Constant in rang of (0,0.5), **Defualt=10^-4

        c2 : Float
          Wolfe constant in rang of (c1,1), **Defualt=0.9 (Recommended Value if 
          Newton or Quasi-Newton method will be used, In case of using Conjugate Gradient 
          recommended value is 0.1 )

        rho: Float
           Scaling factor to reduce alpha, **Defualt=0.1

        alpha0 : Float, **Defualt alpha0=1
            Initial value for alpha
        grad_func : Function 
                Gradient of Objective function , Optional
                **Defualt: if not given will be get from opt_problem    

              

        Outputs :
        ------------
        alpha : Float
           Step Value satisfying Strong Wolfe Conditon 
        iter : INT
           number of Iterations
        '''
        #Definign variables
        phi_prime = grad_func(x + alpha * s) * s.T
        phi0 = grad_func(x) * s.T
        iter = 0
        alpha, _ = self.backtracking_armijo(grad, x, s, c1, rho, alpha0,
                                            max_iter)

        #Loop until we find a alpha
        while not (abs(phi_prime) <= c2 * abs(phi0)) and iter <= max_iter:
            alpha0 *= 1.1
            alpha, _ = self.backtracking_armijo(grad, x, s, c1, rho, alpha0,
                                                max_iter)
