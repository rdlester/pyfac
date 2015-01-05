# Graph class
import numpy as np
from node import FacNode, VarNode
import pdb

""" Factor Graph classes forming structure for PGMs
    Basic structure is port of MATLAB code by J. Pacheco
    Central difference: nbrs stored as references, not ids
        (makes message propagation easier)
"""
        
class Graph:
    """ Putting everything together
    """
    
    def __init__(self):
        self.var = {}
        self.fac = []
        self.dims = []
        self.converged = False
        
    def addVarNode(self, name, dim):
        newId = len(self.var)
        newVar = VarNode(name, dim, newId)
        self.var[name] = newVar
        self.dims.append(dim)
        
        return newVar
    
    def addFacNode(self, P, *args):
        newId = len(self.fac)
        newFac = FacNode(P, newId, *args)
        self.fac.append(newFac)
        
        return newFac
    
    def disableAll(self):
        """ Disable all nodes in graph
            Useful for switching on small subnetworks
            of bayesian nets
        """
        for k, v in self.var.iteritems():
            v.disable()
        for f in self.fac:
            f.disable()
    
    def reset(self):
        """ Reset messages to original state
        """
        for k, v in self.var.iteritems():
            v.reset()
        for f in self.fac:
            f.reset()
        self.converged = False
    
    def sumProduct(self, maxsteps=500):
        """ This is the algorithm!
            Each timestep:
            take incoming messages and multiply together to produce outgoing for all nodes
            then push outgoing to neighbors' incoming
            check outgoing v. previous outgoing to check for convergence
        """
        # loop to convergence
        timestep = 0
        while timestep < maxsteps and not self.converged: # run for maxsteps cycles
            timestep = timestep + 1
            print timestep
            
            for f in self.fac:
                # start with factor-to-variable
                # can send immediately since not sending to any other factors
                f.prepMessages()
                f.sendMessages()
            
            for k, v in self.var.iteritems():
                # variable-to-factor
                v.prepMessages()
                v.sendMessages()
            
            # check for convergence
            t = True
            for k, v in self.var.iteritems():
                t = t and v.checkConvergence()
                if not t:
                    break
            if t:        
                for f in self.fac:
                    t = t and f.checkConvergence()
                    if not t:
                        break
            
            if t: # we have convergence!
                self.converged = True
        
        # if run for 500 steps and still no convergence:impor
        if not self.converged:
            print "No convergence!"
        
    def marginals(self, maxsteps=500):
        """ Return dictionary of all marginal distributions
            indexed by corresponding variable name
        """
        # Message pass
        self.sumProduct(maxsteps)
        
        marginals = {}
        # for each var
        for k, v in self.var.iteritems():
            if v.enabled: # only include enabled variables
                # multiply together messages
                vmarg = 1
                for i in xrange(0, len(v.incoming)):
                    vmarg = vmarg * v.incoming[i]
            
                # normalize
                n = np.sum(vmarg)
                vmarg = vmarg / n
            
                marginals[k] = vmarg
        
        return marginals
    
    def bruteForce(self):
        """ Brute force method. Only here for completeness.
            Don't use unless you want your code to take forever to produce results.
            Note: index corresponding to var determined by order added
            Problem: max number of dims in numpy is 32???
            Limit to enabled vars as work-around
        """
        # Figure out what is enabled and save dimensionality
        enabledDims = []
        enabledNids = []
        enabledNames = []
        enabledObserved = []
        for k, v in self.var.iteritems():
            if v.enabled:
                enabledNids.append(v.nid)
                enabledNames.append(k)
                enabledObserved.append(v.observed)
                if v.observed < 0:
                    enabledDims.append(v.dim)
                else:
                    enabledDims.append(1)
        
        # initialize matrix over all joint configurations
        joint = np.zeros(enabledDims)
        
        # loop over all configurations
        self.configurationLoop(joint, enabledNids, enabledObserved, [])
        
        # normalize
        joint = joint / np.sum(joint)
        return {'joint': joint, 'names': enabledNames}
    
    def configurationLoop(self, joint, enabledNids, enabledObserved, currentState):
        """ Recursive loop over all configurations
            Used for brute force computation
            joint - matrix storing joint probabilities
            enabledNids - nids of enabled variables
            enabledObserved - observed variables (if observed!)
            currentState - list storing current configuration of vars up to this point
        """
        currVar = len(currentState)
        if currVar != len(enabledNids):
            # need to continue assembling current configuration
            if enabledObserved[currVar] < 0:
                for i in xrange(0,joint.shape[currVar]):
                    # add new variable value to state
                    currentState.append(i)
                    self.configurationLoop(joint, enabledNids, enabledObserved, currentState)
                    # remove it for next value
                    currentState.pop()
            else:
                # do the same thing but only once w/ observed value!
                currentState.append(enabledObserved[currVar])
                self.configurationLoop(joint, enabledNids, enabledObserved, currentState)
                currentState.pop()
                
        else:
            # compute value for current configuration
            potential = 1.
            for f in self.fac:
                if f.enabled and False not in [x.enabled for x in f.nbrs]:
                    # figure out which vars are part of factor
                    # then get current values of those vars in correct order
                    args = [currentState[enabledNids.index(x.nid)] for x in f.nbrs]
                
                    # get value and multiply in
                    potential = potential * f.P[tuple(args)]
            
            # now add it to joint after correcting state for observed nodes
            ind = [currentState[i] if enabledObserved[i] < 0 else 0 for i in range(0, currVar)]
            joint[tuple(ind)] = potential
    
    def marginalizeBrute(self, brute, var):
        """ Util for marginalizing over joint configuration arrays produced by bruteForce
        """
        sumout = range(0, len(brute['names']))
        del sumout[brute['names'].index(var)]
        marg = np.sum(brute['joint'], tuple(sumout))
        return marg / np.sum(marg) # normalize to sum to one
