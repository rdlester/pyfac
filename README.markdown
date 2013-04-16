Python implementation of Sum-product (aka Belief-Propagation) for discrete Factor Graphs.

See [this paper](http://www.comm.utoronto.ca/frank/papers/KFL01.pdf) for more details on the Factor Graph framework and the sum-product algorithm.

Requires NumPy.

To use:

    from Graph import Graph
    import numpy as np
    
    G = Graph()
    
    # add variable nodes
    a = G.addVarNode('a',2)
    b = G.addVarNode('b',3)
    
    # add factors
    # unary factor
    Pa = np.array([[0.3],[0.7]])
    G.addFacNode(Pa, a)
    
    # connecting factor
    Pab = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    G.addFacNode(Pab, a, b)
    
    # factors can connect an arbitrary number of variables
    
    # run sum-product and get marginals for variables
    marg = G.marginals()
    distA = marg['a']
    distB = marg['b']
    
    # reset before altering graph further
    G.reset()
    
    # condition on variables
    G.var['a'].condition(0)
    
    # disable and enable to run sum-product on subgraphs
    G.var['b'].disable()
    G.var['b'].enable()
    G.disableAll()
    # reset automatically enables all variables and removes conditioning
