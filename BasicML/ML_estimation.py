#coding:utf-8
import numpy as np
import sys
from pylab import *
 
#M = 3
def y(x, wlist,M):
    ret = wlist[0]
    for i in range(1, M+1):
        ret += wlist[i] * (x ** i)
    return ret
 
def estimate(xlist, tlist,M):
    A = []
    for i in range(M+1):
        for j in range(M+1):
            temp = (xlist**(i+j)).sum()
            A.append(temp)
    A = array(A).reshape(M+1, M+1)
 
    T = []
    for i in range(M+1):
        T.append(((xlist**i) * tlist).sum())
    T = array(T)
    wlist = np.linalg.solve(A, T)
    return wlist
 
def main(M):
    xlist = np.linspace(0, 1, 10)
    tlist = np.sin(2*np.pi*xlist) + np.random.normal(0, 0.2, xlist.size)
    
    w_ml = estimate(xlist, tlist, M)
    print w_ml
    
    N = len(xlist)
    sums = 0
    for n in range(N):
        sums += (y(xlist[n],w_ml, M) - tlist[n])**2
    beta_inv = 1.0/N * sums

    

    xs = np.linspace(0, 1, 500)
    ideal = np.sin(2*np.pi*xs)         
    means = []
    uppers = []
    lowers = []
    
    for x in xs:
        m = y(x, w_ml, M)
        s = sqrt(beta_inv)
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
    M = input('M: ')
    main(M)
