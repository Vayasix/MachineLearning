import math
import random
 
# MCMC base class
class McmcBase:
    @staticmethod
    def rand_norm(mu, sigma):
        r1 = random.random()
        r2 = random.random()
        z = math.sqrt(-2.0 * math.log(r1)) * math.sin(2.0 * math.pi * r2)
        return mu + sigma * z
 
# Metropolis Hastings
class MH(McmcBase):
    def __init__(self, init_x, inv_dist, prop_sigma):
        self.x        = init_x
        self.inv_dist = inv_dist
        self.sigma    = prop_sigma
 
    def update(self):
        candx = MH.rand_norm(self.x, self.sigma)
        curp  = self.inv_dist(self.x)
        candp = self.inv_dist(candx)
        ratio = candp / curp if curp != 0.0 else 1.0
        alph  = min(1.0, ratio)
        r = random.random()
        self.x = candx if r < alph else self.x
 
# Simulated annealing
class SA(McmcBase):
    def __init__(self, init_x, inv_dist, prop_sigma, T0, rho):
        self.x        = init_x
        self.inv_dist = inv_dist
        self.sigma    = prop_sigma
        self.T        = T0
        self.rho      = rho
 
    def update(self):
        candx = SA.rand_norm(self.x, self.sigma)
        curp  = self.inv_dist(self.x) ** (1.0 / self.T)
        candp = self.inv_dist(candx) ** (1.0 / self.T)
        ratio = candp / curp if curp != 0.0 else 1.0
        alph = min(1.0, ratio)
        r = random.random()
        self.x = candx if r < alph else self.x
        self.T = self.T * self.rho
 
# Histogram
class Hist:
    def __init__(self, minx, maxx, nbins):
        self.minx = minx
        self.maxx = maxx
        self.nbins = nbins
        self.span = (maxx - minx) / nbins
        self.bins = [0] * nbins
 
    def set_value(self, x):
        b = int((x - self.minx) / self.span)
        if b >= 0 and b < self.nbins:
            self.bins[b] += 1
 
    def __getitem__(self, i):
        return self.bins[i]
