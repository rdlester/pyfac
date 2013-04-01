import numpy as np
import pdb

""" Factor Graph classes forming structure for PGMs
    Basic structure is port of MATLAB code by J. Pacheco
    Central difference: nbrs stored as references, not ids
        (makes message propagation easier)
    
    Note to self: use %pdb and %load_ext autoreload followed by %autoreload 2
"""

class Node(object):
    """ Superclass for graph nodes
    """
    epsilon = 10**(-4)
    
    def __init__(self, nid):
        self.enabled = True
        self.nid = nid
        self.nbrs = []
        self.incoming = []
        self.outgoing = []
        self.oldoutgoing = []
    
    def reset(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def enable(self):
        self.enabled = True
        for n in self.nbrs:
            # don't call enable() as it will recursively enable entire graph
            n.enabled = True
    
    def nextStep(self):
        """ Used to have this line in prepMessages
            but it didn't work?
        """
        self.oldoutgoing = self.outgoing[:]
    
    def normalizeMessages(self):
        """ Normalize to sum to 1
        """
        self.outgoing = [x / np.sum(x) for x in self.outgoing]
    
    def receiveMessage(self, f, m):
        """ Places new message into correct location in new message list
        """
        if self.enabled:
            i = self.nbrs.index(f)
            self.incoming[i] = m
    
    def sendMessages(self):
        """ Sends all outgoing messages
        """
        for i in xrange(0, len(self.outgoing)):
            self.nbrs[i].receiveMessage(self, self.outgoing[i])
    
    def checkConvergence(self):
        """ Check if any messages have changed
        """
        if self.enabled:
            for i in xrange(0, len(self.outgoing)):
                # check messages have same shape
                self.oldoutgoing[i].shape = self.outgoing[i].shape
                delta = np.absolute(self.outgoing[i] - self.oldoutgoing[i])
                if (delta > Node.epsilon).any(): # if there has been change
                    return False
            return True
        else:
            # Always return True if disabled to avoid interrupting check
            return True

class VarNode(Node):
    """ Variable node in factor graph
    """
    def __init__(self, name, dim, nid):
        super(VarNode, self).__init__(nid)
        self.name = name
        self.dim = dim
        self.observed = -1 # only >= 0 if variable is observed
    
    def reset(self):
        super(VarNode, self).reset()
        size = range(0, len(self.incoming))
        self.incoming = [np.ones((self.dim,1)) for i in size]
        self.outgoing = [np.ones((self.dim,1)) for i in size]
        self.oldoutgoing = [np.ones((self.dim,1)) for i in size]
        self.observed = -1
    
    def condition(self, observation):
        """ Condition on observing certain value
        """
        self.enable()
        self.observed = observation
        # set messages (won't change)
        for i in xrange(0, len(self.outgoing)):
            self.outgoing[i] = np.zeros((self.dim,1))
            self.outgoing[i][self.observed] = 1.
        self.nextStep() # copy into oldoutgoing
    
    def prepMessages(self):
        """ Multiplies together incoming messages to make new outgoing
        """
        
        # compute new messages if no observation has been made
        if self.enabled and self.observed < 0 and len(self.nbrs) > 1:
            # switch reference for old messages
            self.nextStep()
            for i in xrange(0, len(self.incoming)):
                # multiply together all excluding message at current index
                curr = self.incoming[:]
                del curr[i]
                self.outgoing[i] = reduce(np.multiply, curr)
        
            # normalize once finished with all messages
            self.normalizeMessages()

class FacNode(Node):
    """ Factor node in factor graph
    """
    def __init__(self, P, nid, *args):
        super(FacNode, self).__init__(nid)
        self.P = P
        self.nbrs = list(args) # list storing refs to variable nodes
        
        # num of edges
        numNbrs = len(self.nbrs)
        numDependencies = self.P.squeeze().ndim
        
        # init messages
        for i in xrange(0,numNbrs):
            v = self.nbrs[i]
            vdim = v.dim
            
            # init for factor
            self.incoming.append(np.ones((vdim,1)))
            self.outgoing.append(np.ones((vdim,1)))
            self.oldoutgoing.append(np.ones((vdim,1)))
            
            # init for variable
            v.nbrs.append(self)
            v.incoming.append(np.ones((vdim,1)))
            v.outgoing.append(np.ones((vdim,1)))
            v.oldoutgoing.append(np.ones((vdim,1)))
        
        # error check
        assert (numNbrs == numDependencies), "Factor dimensions does not match size of domain."
    
    def reset(self):
        super(FacNode, self).reset()
        for i in xrange(0, len(self.incoming)):
            self.incoming[i] = np.ones((self.nbrs[i].dim,1))
            self.outgoing[i] = np.ones((self.nbrs[i].dim,1))
            self.oldoutgoing[i] = np.ones((self.nbrs[i].dim,1))
    
    def prepMessages(self):
        """ Multiplies incoming messages w/ P to make new outgoing
        """
        if self.enabled:
            # switch references for old messages
            self.nextStep()
        
            mnum = len(self.incoming)
        
            # do tiling in advance
            # roll axes to match shape of newMessage after
            for i in xrange(0,mnum):
                # find tiling size
                nextShape = list(self.P.shape)
                del nextShape[i]
                nextShape.insert(0, 1)
                # need to expand incoming message to correct num of dims to tile properly
                prepShape = [1 for x in nextShape]
                prepShape[0] = self.incoming[i].shape[0]
                self.incoming[i].shape = prepShape
                # tile and roll
                self.incoming[i] = np.tile(self.incoming[i], nextShape)
                self.incoming[i] = np.rollaxis(self.incoming[i], 0, i+1)
            
            # loop over subsets
            for i in xrange(0, mnum):
                curr = self.incoming[:]
                del curr[i]
                newMessage = reduce(np.multiply, curr, self.P)
                    
                # sum over all vars except i!
                # roll axis i to front then sum over all other axes
                newMessage = np.rollaxis(newMessage, i, 0)
                newMessage = np.sum(newMessage, tuple(range(1,mnum)))
                newMessage.shape = (newMessage.shape[0],1)
                    
                #store new message
                self.outgoing[i] = newMessage
        
            # normalize once finished with all messages
            self.normalizeMessages()