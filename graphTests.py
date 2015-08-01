from __future__ import print_function
from graph import Graph
import numpy as np

""" Graphs for testing sum product implementation
"""

def checkEq(a,b):
    epsilon = 10**-6
    return abs(a-b) < epsilon

def makeToyGraph():
    """ Simple graph encoding, basic testing
        2 vars, 2 facs
        f_a, f_ba - p(a)p(a|b)
        factors functions are a little funny but it works
    """
    G = Graph()

    a = G.addVarNode('a', 3)
    b = G.addVarNode('b', 2)

    Pb = np.array([[0.3], [0.7]])
    G.addFacNode(Pb, b)

    Pab = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    G.addFacNode(Pab, a, b)

    return G

def testToyGraph():
    """ Actual test case
    """

    G = makeToyGraph()
    marg = G.marginals()
    brute = G.bruteForce()

    # check the results
    # want to verify incoming messages
    # if vars are correct then factors must be as well
    a = G.var['a'].incoming
    assert checkEq(a[0][0], 0.34065934)
    assert checkEq(a[0][1], 0.2967033)
    assert checkEq(a[0][2], 0.36263736)

    b = G.var['b'].incoming
    assert checkEq(b[0][0], 0.3)
    assert checkEq(b[0][1], 0.7)
    assert checkEq(b[1][0], 0.23333333)
    assert checkEq(b[1][1], 0.76666667)


    # check the marginals
    am = marg['a']
    assert checkEq(am[0], 0.34065934)
    assert checkEq(am[1], 0.2967033)
    assert checkEq(am[2], 0.36263736)

    bm = marg['b']
    assert checkEq(bm[0], 0.11538462)
    assert checkEq(bm[1], 0.88461538)

    # check brute force against sum-product
    amm = G.marginalizeBrute(brute, 'a')
    bmm = G.marginalizeBrute(brute, 'b')
    assert checkEq(am[0], amm[0])
    assert checkEq(am[1], amm[1])
    assert checkEq(am[2], amm[2])
    assert checkEq(bm[0], bmm[0])
    assert checkEq(bm[1], bmm[1])

    print("All tests passed!")

def makeTestGraph():
    """ Graph for HW problem 1.c.
        4 vars, 3 facs
        f_a, f_ba, f_dca
    """
    G = Graph()

    a = G.addVarNode('a', 2)
    b = G.addVarNode('b', 3)
    c = G.addVarNode('c', 4)
    d = G.addVarNode('d', 5)

    p = np.array([[0.3], [0.7]])
    G.addFacNode(p, a)

    p = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    G.addFacNode(p, b, a)

    p = np.array([ [[3., 1.], [1.2, 0.4], [0.1, 0.9], [0.1, 0.9]], [[11., 9.], [8.8, 9.4], [6.4, 0.1], [8.8, 9.4]], [[3., 2.], [2., 2.], [2., 2.], [3., 2.]], [[0.3, 0.7], [0.44, 0.56], [0.37, 0.63], [0.44, 0.56]], [[0.2, 0.1], [0.64, 0.44], [0.37, 0.63], [0.2, 0.1]] ])
    G.addFacNode(p, d, c, a)

    # add a loop - not a part of 1.c., just for testing
    # p = np.array([[0.3, 0.2214532], [0.1, 0.4] , [0.33333, 0.76], [0.1, 0.98]])
#     G.addFacNode(p, c, a)

    return G

def testTestGraph():
    """ Automated test case
    """
    G = makeTestGraph()
    marg = G.marginals()
    brute = G.bruteForce()

    # check the marginals
    am = marg['a']
    assert checkEq(am[0], 0.13755539)
    assert checkEq(am[1], 0.86244461)

    bm = marg['b']
    assert checkEq(bm[0], 0.33928227)
    assert checkEq(bm[1], 0.30358863)
    assert checkEq(bm[2], 0.3571291)

    cm = marg['c']
    assert checkEq(cm[0], 0.30378128)
    assert checkEq(cm[1], 0.29216947)
    assert checkEq(cm[2], 0.11007584)
    assert checkEq(cm[3], 0.29397341)

    dm = marg['d']
    assert checkEq(dm[0], 0.076011)
    assert checkEq(dm[1], 0.65388724)
    assert checkEq(dm[2], 0.18740039)
    assert checkEq(dm[3], 0.05341787)
    assert checkEq(dm[4], 0.0292835)

    # check brute force against sum-product
    amm = G.marginalizeBrute(brute, 'a')
    bmm = G.marginalizeBrute(brute, 'b')
    cmm = G.marginalizeBrute(brute, 'c')
    dmm = G.marginalizeBrute(brute, 'd')

    assert checkEq(am[0], amm[0])
    assert checkEq(am[1], amm[1])

    assert checkEq(bm[0], bmm[0])
    assert checkEq(bm[1], bmm[1])
    assert checkEq(bm[2], bmm[2])

    assert checkEq(cm[0], cmm[0])
    assert checkEq(cm[1], cmm[1])
    assert checkEq(cm[2], cmm[2])
    assert checkEq(cm[3], cmm[3])

    assert checkEq(dm[0], dmm[0])
    assert checkEq(dm[1], dmm[1])
    assert checkEq(dm[2], dmm[2])
    assert checkEq(dm[3], dmm[3])
    assert checkEq(dm[4], dmm[4])

    print("All tests passed!")

# standard run of test cases
testToyGraph()
testTestGraph()
