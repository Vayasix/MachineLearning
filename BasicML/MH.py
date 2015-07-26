import matplotlib.pyplot as plt
import mcmc
from math import *
 
# invariant distribution
def inv_dist(x):
    r = 0.3
    mu1 = 1.0
    mu2 = -2.0
    x1 = x - mu1
    x2 = x - mu2
    e1 = exp(-x1 * x1) / sqrt(pi)
    e2 = exp(-x2 * x2) / sqrt(pi)
    return r * e1 + (1.0 - r) * e2
 
def mh_test(init_x, trial, inv_dist):
    # instantiate M-H object
    burn = int(trial / 10)
    mh = mcmc.MH(init_x, inv_dist, 1.0)
 
    # instantiate histogram object
    minx = -5.0
    maxx =  5.0
    nbins = 40
    hist = mcmc.Hist(minx, maxx, nbins)
 
    # burn-in
    for i in range(burn):
        mh.update()
 
    # M-H simulation
    for i in range(trial):
        hist.set_value(mh.x)
        mh.update()
 
    # show histogram
    ys = [hist[i] / (trial * hist.span) for i in range(nbins)]
    xs = [hist.span * i + minx for i in range(nbins)]
    plt.bar(xs, ys, width = hist.span)
 
    # show invariant distributin
    ndiv = 400
    span = (maxx - minx) / ndiv
    xs = [span * i + minx for i in range(ndiv)]
    zs = [inv_dist(xs[i]) for i in range(ndiv)]
    plt.plot(xs, zs, 'r-')
    plt.show()
 
if __name__=='__main__':
    mh_test(0.0, 100000, inv_dist)

