#coding:utf-8
import numpy as np
from pylab import *
import sys

M = 9
ALPHA = 0.005
BETA = 11.1

def y(x, wlist):
    ret = wlist[0]
    for i in range(1, M+1):
        ret += wlist[i] * (x ** i)
    return ret

def phi(x):
    data = []
    for i in range(0, M+1):
        data.append(x**i)
    ret = np.matrix(data).reshape((M+1, 1))  
    return ret

def mean(x, xlist, tlist, S):
    sums = matrix(zeros((M+1, 1)))
    for n in range(len(xlist)):
        sums += phi(xlist[n]) * tlist[n]
    ret = BETA * phi(x).transpose() * S * sums
    return ret

def variance(x, xlist, S):
    ret = 1.0 / BETA + phi(x).transpose() * S * phi(x)
    return ret

def main():
    # train data
    # sin(2*pi*x)+random_noise from gausian distribution
    xlist = np.linspace(0, 1, 10)
    tlist = np.sin(2*np.pi*xlist) + np.random.normal(0, 0.2, xlist.size)
    
    # bayes curve fitting-> evaluate predictive distribution
    # calc Matrix:S
    sums = matrix(zeros((M+1, M+1)))
    for n in range(len(xlist)):
        sums += phi(xlist[n]) * phi(xlist[n]).transpose()
    I = matrix(np.identity(M+1))
    S_inv = ALPHA * I + BETA * sums
    S = S_inv.getI()

    xs = np.linspace(0, 1, 500)
    ideal = np.sin(2*np.pi*xs) #ideal curve
    means = []
    uppers = []
    lowers = []
    for x in xs:
        m = mean(x, xlist, tlist, S)[0,0]       
        s = np.sqrt(variance(x, xlist, S)[0,0])
        u = m + s                             
        l = m - s                            
        means.append(m)
        uppers.append(u)
        lowers.append(l)
    
    plot(xlist, tlist, 'bo')  
    plot(xs, ideal, 'g-')     
    plot(xs, means, 'r-')     
    plot(xs, uppers, 'r--')  
    plot(xs, lowers, 'r--') 
    xlim(0.0, 1.0)
    ylim(-1.5, 1.5)
    show()

if __name__ == "__main__":
    main()


