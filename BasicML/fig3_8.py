#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

ALPHA = 2.0
BETA = 25
M = 9

def phi(x):
    data = []
    for n in range(M+1):
        data.append(x**i)
    ret = np.array(data).reshape(M+1,1)
    return ret

def mean():

def variance(x,S):
    ret = 1/BETA + phi(x).transpose() * S * phi(x)+1
    return ret

def main():
    N = 1
    #train data
    xlist = np.linspace(0,1, N)
    tlist = np.sin(2*np.pi*xlist)+np.random.normal(0,0.2,xlist.size)
    
    xs = np.linspace(0,1,1000)
    ideal = np.sin(2*np.pi*xs)
    
    #予測分布を求める
    #行列Sを計算


if __name__ == "__name__":
    main()

