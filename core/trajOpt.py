# coding: utf-8
'''
Trajectory Optimization
Developer:
-----------
Kurosh Zamani

-----------
ToDo :

'''
#from __future__ import deviation, print_function
#=============================================================
# Standard Python modules
#=============================================================
import os, sys
import types
import logging


#=============================================================
# External Python modules
#=============================================================
import numpy as np
from scipy.optimize import minimize
import scipy.io as sio


# from numdifftools import Jacobian
import ipydex


class TrajProblem(object):
    """
    This is a Container for defining trajectory problem

    Inputs :
        ---------
        guess : dict
               dict of guess values with :
               keys  :'finalTime', 'initialState','finalState', 'initialControl', 'finalControl'
               values: list

        bounds : dict
                bounds for the trajecotry problem with:
                keys : 'initialState', 'finalState', 'state', 'initialControl','finalControl', 'control', 'finatlTime'
                values : dicts each of which has 'Low' and 'Upp' keys .they are determining
                         lower and Upper bounds

        dynamics : numpy function
                should returns state function x_dot= F(x,u)

        pathObjective: numpy function
                should retruns Integrand of pathObjective
    """

    def __init__(self,
                 guess,
                 dynamics,
                 pathObjective,
                 bounds=None,
                 label=None,
                 **kwargs):

        self.pathObj= pathObjective
        self.dynamics= dynamics
        # cast the required keys for guess:
        self.guess = {
            'finalTime': None,
            'initialState': None,
            'finalState': None,
            'initialControl': None,
            'finalControl': None
        }

        # ckeck the guess :
        if isinstance(guess, dict):
            self._checkInput(guess, self.guess, eqCheck=True)
            self.guess.update(guess)

        # cast the required keys for bounds :
        nState=self.nState=len(self.guess['initialState'])
        nControl=self.nControl=len(self.guess['initialControl'])

        bnds_initialState = {'Upp': [None for i in range(nState)], 'Low': [None for i in range(nState)]}
        bnds_finalState =  {'Upp': [None for i in range(nState)], 'Low': [None for i in range(nState)]}
        bnds_state = {'Upp': [None for i in range(nState)], 'Low': [None for i in range(nState)]}
        bnds_initialControl = {'Upp': [None], 'Low': [None]}
        bnds_finalControl = {'Upp': [None], 'Low': [None]}
        bnds_control = {'Upp': [None], 'Low': [None]}
        bnds_initialTime= {'Upp': 0.0, 'Low': 0.0}
        bnds_finalTime = {'Upp': 0.1+self.guess['finalTime'], 'Low':self.guess['finalTime']-0.1}

        self.bounds = {
            'initialState': bnds_initialState,
            'finalState': bnds_finalState,
            'state': bnds_state,
            'initialControl': bnds_initialControl,
            'finalControl': bnds_finalControl,
            'control': bnds_control,
            'initialTime' : bnds_initialTime,
            'finalTime': bnds_finalTime
        }

        #check the bounds :
        if isinstance(bounds, dict):
            self._checkInput(bounds, self.bounds)
            for bndKey, bndVal in bounds.items():
                # check if we got a dict as bound
                assert isinstance(bndVal,dict), '{} in bounds should be a dictionary!'.format(bndKey)

                # check if the keys are match the required keys for bounds
                self._checkInput(bndVal,self.bounds[bndKey],keyLabel='bounds' + str([bndKey]))
                self.bounds[bndKey].update(bndVal)

    def _checkInput(self, inputdict, mydict, keyLabel=None, eqCheck=False):
        '''
        '''
        if eqCheck:
            a = set(list(inputdict.keys()))
            b = set(list(mydict.keys()))
            assert len(a.intersection(b)) == len(a.union(
                b)), " {} in {}  aren't match all the required {}! ".format(
                    inputdict.keys(), 'guess', mydict.keys())

        else:

            assert set(list(inputdict.keys())).issubset(
                set(list(mydict.keys()))
            ), " {} in {}  aren't match the required {}! ".format(
                inputdict.keys(), keyLabel, mydict.keys())

        return


class TrajOptimization(object):
    """
        This class provides trajectory optimization with Direct Collocation method .

    """

    def __init__(self, problem=None, dynamics=None):
        """
        Inputs :
        -------
        problem : an instance of a Calss TrajProblem

        """
        # assert isinstance(
        #     problem, TrajProblem
        # ), "problem should be an instance of the Class TrajProblem!"
        if isinstance(problem, TrajProblem):
            self.problem = problem
        self.guess = {'time': [], 'state': [], 'control': []}
        
        if isinstance(dynamics, types.FunctionType):
            self.dynamics=dynamics
        # self.solutions= {}

    def hermitsimpson(self, nSegment):
        """
        """

        # each segment needs an additional data point in the middle , thus :
        nGrid = self.nGrid= 2 * nSegment + 1
        self.nSegment= nSegment
        self.SimpWeights = (2.0 / 3) * np.ones(nGrid)
        self.SimpWeights[1::2] = 4.0 / 3
        self.SimpWeights[0] = 1.0 / 3
        self.SimpWeights[-1] = 1.0 / 3

        # self.problem.hsDefectCst = self.computeHsDefects
        self.directCollocation(self.problem)

        # interpolation the results
        tSol=self.tSol
        uSol=self.uSol
        xSol=self.xSol
        self.fSol= self.problem.dynamics(tSol, xSol, uSol)
       
        return

    def directCollocation(self, problem):
        """
        """
        prob=self.problem
        nGrid=self.nGrid

        # Interpolate the guess at the grid points for transcription
        self.guessInterp(nGrid)

        # Unpack all the boudns:
        tLow= [prob.bounds['initialTime']['Low'], prob.bounds['finalTime']['Low']]
        tUpp= [prob.bounds['initialTime']['Upp'], prob.bounds['finalTime']['Upp']]
        tBounds=self.tBounds=list(zip(tLow, tUpp))  # list is just a trick to scape a Bug in ipython!

        xLow= prob.bounds['initialState']['Low']+ \
         np.array([bnd for i in range(nGrid-2) for bnd in prob.bounds['state']['Low']]).tolist()+ \
         prob.bounds['finalState']['Low']

        xUpp= prob.bounds['initialState']['Upp']+ \
         np.array([bnd for i in range(nGrid-2) for bnd in prob.bounds['state']['Upp']]).tolist()+ \
         prob.bounds['finalState']['Upp']

        self.xBounds=xBounds=list(zip(xLow, xUpp))

        uLow= prob.bounds['initialControl']['Low'] + \
         np.array([bnd for i in range(nGrid-2) for bnd in prob.bounds['control']['Low']]).tolist() +\
         prob.bounds['finalControl']['Low']

        uUpp= prob.bounds['initialControl']['Upp'] + \
         np.array([bnd for i in range(nGrid-2) for bnd in prob.bounds['control']['Upp']]).tolist() +\
         prob.bounds['finalControl']['Upp']

        uBounds=self.uBounds=list(zip(uLow, uUpp))
        # Pack to zBound
        zBounds = tBounds
        nState = prob.nState
        nControl = prob.nControl
        for i in range(nGrid):
            zBounds= zBounds + xBounds[i*nState:(i+1)*nState] + uBounds[i*nControl:(i+1)*nControl]

        self.zBounds=zBounds
        # solve the optimization problem
        cons={'type': 'eq', 'fun': self.hsEqConstraint}

        print("===========================================")
        print("Trajectory optimization with {} Segments, Iter:{}".format(self.nSegment, self.iterNum))
        self.opt_result = minimize(
            self.costFunc,
            self.zGuess,
            method='SLSQP',
            constraints=cons,
            bounds=zBounds,
            options={
                'ftol': 0.001,
                'disp': True
            })

        zSol= self.opt_result.x
        tSol, xSol, uSol = self.unPackDecVar(zSol)
        self.tSol=tSol
        self.xSol=xSol
        self.uSol=uSol



        return

    def guessInterp(self, nGrid):
        '''
        Interpolates to find zGuess needed for  scipy minimize
        by first guess it interpolates between first and last gueses 
        given from user. after that it uses results from last optimization !
    
        '''

        prob = self.problem
        tf = prob.guess['finalTime']


        if self.iterNum == 1 :

            # time
            time = [0, tf]

            # state
            initialState = prob.guess['initialState']
            finalState = prob.guess['finalState']
            state = np.array([initialState, finalState])

            #control
            initialControl = prob.guess['initialControl']
            finalControl = prob.guess['finalControl']
            control =np.array([initialControl, finalControl])

        else:
            time= self.tSol
            state= self.xSol
            control= self.uSol

        newTime = self.guess['time'] = np.linspace(0, tf, nGrid)
        self.guess['state'] = self._interp2(newTime, time, state)
        self.guess['control'] = self._interp2(newTime, time, control)

        self.zGuess = self.packDecVar(**self.guess)

    def _interp1(self, tSpan, gTime, yy):
        """
        interpolate between inital and final values
        as a first guess for states
        """

        ta = tSpan[0]
        tb = tSpan[1]
        nyy = len(yy[0])
        nGrids = self.nGrid

        xx = list(
            zip(ta * np.ones(nyy), tb * np.ones(nyy))
        )  # adding list to zip is just a trick to escape the bug in ipython

        yy = list(zip(yy[0], yy[1]))

        shape = (nGrids, nyy)

        ret = np.zeros(shape)
        idx = 0
        for x, y in list(zip(xx, yy)):
            ret[:, idx] = np.interp(gTime, x, y)
            idx += 1

        return ret

    def _interp2(self, newTime, time, yy):
        """
        given a set of time,yy it interpolates yy in newTime 
        
        """


        nyy = len(yy[0])
        nGrid= len(newTime)

        shape = (nGrid, nyy)
        ret = np.zeros(shape)

        for idx in range(nyy) :
            y= yy[:, idx]
            ret[:, idx]= np.interp(newTime, time, y)

        return ret

    def packDecVar(self, time, state, control):
        """
        This function collapses the time, state and control
        into a single vector

        Inputs:
        -------
        time=  time vector (grid points)
        state=  state vector at each grid point, shape :(nState, nTime)
        control=  contro vector at each grid point, shape: (nControl, nTime)

        Returns:
        ------
        z=   Vector of 2+ nTime*(nState+nControl) decision variables
        """
        tt = time
        xx = state
        uu = control

        nTime = self.nTime = len(tt)
        nState = self.nState = xx.shape[1]
        nControl = self.nControl = uu.shape[1]

        tSpan = [tt[0], tt[-1]]

        xVec = xx.ravel()
        uVec = uu.ravel()
        n = len(xVec) + len(uVec)
        indz = np.arange(2, n + 2).reshape(nTime, nState + nControl)
        m = len(indz.ravel())

        # index of time, state, control variables in the decVar vector
        tIdx = np.arange(0, 2)
        xIdx = indz[:, 0:nState].ravel()
        uIdx = indz[:, nState:nControl + nState].ravel()

        z = np.zeros(2 + m)
        z[tIdx[:]] = tSpan
        z[xIdx[:]] = xVec
        z[uIdx[:]] = uVec

        self.tIdx = tIdx
        self.xIdx = xIdx
        self.uIdx = uIdx

        return z

    def hsEqConstraint(self, z):
        """
        unpacks the decision variables, compute defects along the trajectory, and
        evalutes the userd_defined constraint function
        """
        tt, xx, uu= self.unPackDecVar(z)

        dt=float(tt[-1]-tt[0])/(self.nTime-1)
        ff=self.problem.dynamics(tt, xx, uu)

        # Compute defects along the trajecotry
        defects= self.computeHsDefects(dt, xx, ff )
        #TODO: add user_defined constraints and pack all Csts up

        return defects



    def computeHsDefects(self, dt, xx, ff, dtGrad=None, xGrad=None, ffGrad=None ):
        """
        coputes the defects that are used to enforce the continous dynamics of the system along the trajectory

        Inputs:
        -------
        dt = time step (scalar)
        xx = state at eatch grid point
        ff = dynamics of the state along the trajectory
        dtGrad = gradient of time step wrt [t0, tf]
        xGrad= gradient of the trajecotry wrt dec vars
        ffGrad= gradient of dynamics wrt dec vars

        Retruns :
        -------
        defects = error in dynamics along the trajectory
        defectsGrad= gradient of defects
        """
        nTime=self.nTime
        nState=self.nState

        iLow=np.arange(0, nTime-1, 2)
        iMid= iLow+ 1
        iUpp= iMid+1

        xxLow= xx[iLow, :]
        xxMid= xx[iMid, :]
        xxUpp= xx[iUpp, :]

        ffLow= ff[iLow, :]
        ffMid= ff[iMid, :]
        ffUpp= ff[iUpp, :]

        # Mid point Constraints (Hermite)
        defectMidpoint= xxMid - (xxUpp + xxLow)/2. - dt*(ffLow - ffUpp)/4.

        # Interval Constraints (Simpson)
        defectInterval= xxUpp - xxLow - dt*(ffUpp + 4.0*ffMid +ffLow)/3.

        # pack everythings up
        defects=np.zeros((nTime-1, nState))
        defects[iLow,:]= defectInterval
        defects[iMid,:]= defectMidpoint

        # Gradient Calculations:
        # TODO : Write Gradient

        return defects.ravel()


    def costFunc(self, z):
        """
        unpacks the decision variables, send them to the user_defined objective functions,
        and then returns the final cost.

        """
        # unpack time, state and Control
        tt, xx, uu= self.unPackDecVar(z)

        # Compute the Cost Integral along trajectory
        dt= float(tt[-1]-tt[0])/(self.nTime-1)
        integrand=self.problem.pathObj(tt, xx, uu)
        intCost= dt*integrand*self.SimpWeights

        integralCost=0.0
        for cost in intCost:
            integralCost+=cost


        # TODO: add the cost at boundries of the trajectory

        return integralCost



    def unPackDecVar(self, z):
        """
        unpacks the decision variables for trajectroy optimization into the
        time, state and control matriceis
        """

        nTime = self.nTime
        nState = self.nState
        nControl = self.nControl

        tt= np.linspace(z[0], z[1], nTime)

        xx = z[self.xIdx]
        uu = z[self.uIdx]

        # reshape xx and uu (nTime, nState)
        xx= xx.reshape(nTime, nState)
        uu= uu.reshape(nTime, nControl)

        return tt, xx, uu


    def solve(self, nSegments=[12], method='hermitSimpson'):
        '''
        solves Trajecotry problem with different methods
        (curently just hermitSimpson!)
        
        Inputs:
        -----
        nSegments : list
                    a list containing number of segments for each Iteration
        '''
        self.iterNum=1
        if method == 'hermitSimpson':
            for nSegment in nSegments:
                self.hermitsimpson(nSegment)
                self.iterNum+=1


    def pwPoly3(self, tGrid, xGrid, fGrid, t):
        """
        Returns pice-wise quadratic interpolation of a set of data,
        given the function value at the edge and midpoint of the
        interval of interest.
        

        Inputs:
        ---------

        """
        nGrid= len(tGrid)
        n= int((nGrid-1)/2)
        m= xGrid.shape[1]
        k= len(t)
        x= np.zeros((k, m))
        # Figure out which segment each value of t should be on
        edges= np.array([-np.inf] + tGrid[::2].tolist()+ [np.inf])
        bins = np.digitize(t, edges) -1

        # loop over each qudratic segment
        for i in range(1,n+1):
            mask = bins == i
            if np.sum(mask) > 0 :
                # convert logical array idx to indx
                idx= self.findIndex(mask)

                # find Grid points needed
                kLow= 2*(i-1)
                kMid= kLow +1
                kUpp= kLow +2
                h= tGrid[kUpp]-tGrid[kLow]
                xLow= xGrid[kLow, :]
                fLow= fGrid[kLow, :]
                fMid= fGrid[kMid, :]
                fUpp= fGrid[kUpp, :]
                alpha= t[idx] - tGrid[kLow]
                x[idx, :]= self.cubicInterp(h, xLow, fLow, fMid, fUpp, alpha)
    
        #TODO:check if there is a t out of range of expected !

        return x


    def findIndex(self, mask):
        '''
        
        '''
        indices= mask*np.arange(1,len(mask)+1)
        indices=[index for index in indices if index > 0]
        ret =np.array(indices)-1

        return ret


    def cubicInterp(self, h, xLow, fLow, fMid, fUpp, alpha):
        '''
        Returns interpolant over a single interval

        Inputs:
        -----
        h    : time step
        xLow : function value at tLow
        fLow : derivative at tLow
        fMid : derivative at tMid
        fUpp : derivative at tUpp
        alpha: query points on domain [0, h]

        Outputs:
        -----
        x    : (m, p) function at query times

        '''
        # Fix matrix dimensions for vectorized calculations
        nx= len(xLow)
        nt= len(alpha)
        xLow= xLow*np.ones((nt, 1))
        fLow= fLow*np.ones((nt, 1))
        fMid= fMid*np.ones((nt, 1))
        fUpp= fUpp*np.ones((nt, 1))
        alpha=alpha* np.ones((nx,1))

        a = (2.0 * (fLow - 2.0 * fMid + fUpp)) / (3 * h**2)
        b = -(3 * fLow - 4.0 * fMid + fUpp) / (2 * h)
        c = fLow
        d = xLow

        x = d + alpha.T * (c + alpha.T * (b + alpha.T * a))
        return x


    def pwPoly2(self,tGrid, xGrid, t):
        '''
        Returns pice-wise equdratic interpolation of ta set of data, 
        given the function value at the edges and midpoint of the interval
        of interest.
        '''
        nGrid=len(tGrid)

        n= int((nGrid-1)/2)
        m= xGrid.shape[1]
        k= len(t)
        xx=np.zeros((k, m))

        #Figure aout which segment each value of t should be on
        edges= np.array([-np.inf] + tGrid[::2].tolist()+ [np.inf])
        bins = np.digitize(t, edges) -1

        # Loop over each quadratic segment
        for i in range(1, n+1):
            mask= bins == i
            if np.sum(mask) > 0:
                idx=self.findIndex(mask)
                gridIdx= np.array([0, 1, 2])+ 2*(i-1)
                xx[idx, :]=self.quadInterp(tGrid[gridIdx], xGrid[gridIdx, :], t[idx])
                
        return xx



    def quadInterp(self, tGrid, xGrid, t):
        '''
        Inputs :
        ------
        tGrid :  time grid
        xGrid :  function grid
        t     :  query times, spanned by tGrid
        '''
        # Rescale the query points to be on the domain [-1 1]
        t = 2 * (t - tGrid[0]) / (tGrid[2] - tGrid[0]) - 1

        # Compute the coefficients:
        a = 0.5 * (xGrid[2, :] + xGrid[0, :]) - xGrid[1, :]
        b = 0.5 * (xGrid[2, :] - xGrid[0, :])
        c = xGrid[1, :]

        # Evaluate the polynomial for each dimension of the function
        p = len(t)
        m = xGrid.shape[1]
        # x = np.zeros((p, m))
        tt= t**2

        xx= a*tt + b*t + c
        
        return xx.reshape(p, m)
    
    def loadMatlabResults(self, fileName):
        '''
        Loads results saved in .mat format
        '''
        current_path = os.getcwd()
        current_dir = current_path.split('/')[-1]
        try:

            if current_dir == 'core':
                os.chdir('../MatRes')
            elif current_dir == 'triple_pendulum':
                os.chdir('MatRes')
        except:
            print('filename could not be found! you have to run the code in /triple_pendulum or /core Folder! the Matlab Files should be in /Triple_pendulum/MatRes')
            raise

        content=sio.loadmat(fileName)
        time= content['time'].ravel()
        n=len(time)
        control=content['control'].reshape(n, 1)
        state= content['state'].reshape(n, 8)
        self.matState=state
        self.matControl=control
        self.matTime=time

        os.chdir(current_path)

        return time, state, control

    def convertMatResToFunc(self, fileName):
        '''
        Loads grids from matlab and converts 
        them to control and state functions
        it assumes that the resuls are saved with 
        enought resolutions so it uses linear interpolation
        '''
        time, state, control= self.loadMatlabResults(fileName)

        def stateFunc(t):
            ret=self._interp2(np.array([t]), time, state)
            return ret

        def controlFunc(t):
            ret=self._interp2(np.array([t]), time, control)
            return ret    

        return stateFunc, controlFunc  

    def convertGridsToFunc(self, fileName=None):
        '''
        Loads grids (from matlab if a fileName is given else form python results)and converts 
        them to control and state functions
        (it uses quadratic interpolation for control 
        and cubic interpolation for state) 
        you sould also give dynamics function
        in your object definition !

        '''
        if fileName:
            tSol, xSol, uSol= self.loadMatlabResults(fileName)
            fSol= self.dynamics(tSol, xSol, uSol)
        
        else:
            tSol=self.tSol
            uSol=self.uSol
            xSol=self.xSol
            fSol=self.fSol


        def stateFunc(t):

            # flag=False    
            # if isinstance(t, (float, int)):
            #     tt=np.array([t])
            #     flag=True

            # elif isinstance(t, list):
            #     tt=np.array(t)    

            # elif isinstance(t, np.ndarray) :
            #     tt=t
            tt=np.array([t])
            ret=self.pwPoly3( tSol, xSol, fSol, tt )
            # if flag:
            #     ret=ret.ravel()
            return ret.ravel()

        def controlFunc(t):
            # flag=False
            # if isinstance(t, (float, int)):
            #     tt=np.array([t])
            #     flag=True
            # elif isinstance(t, list):
            #     tt=np.array(t)    

            # elif isinstance(t, np.ndarray) :
            #     tt=t
            tt=np.array([t])
            ret=self.pwPoly2( tSol, uSol, tt)
            
            return ret.ravel()    

        return stateFunc, controlFunc
        
    def collCst(self, t, xFunc, uFunc):
        '''
        collocation cosntraint= dynamics - (derivative of state trajectroy)
        '''
        xx=xFunc(t)
        uu=uFunc(t)
        fSol=self.dynamics(self.matTime, self.matState, self.matControl)
        ret= self.dynamics(t, xx, uu )- self.pwPoly2(self.matTime, fSol, t)
        return ret