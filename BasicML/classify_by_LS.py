#coding:utf-8
# 4.1.3 分類における最小二乗（p.182）
import numpy as np
from pylab import *
import sys

K = 2    # 2classes
N = 100  # Num.data

def f(x1, W_t):
    a = - ((W_t[0,1]-W_t[1,1]) / (W_t[0,2]-W_t[1,2]))
    b = - (W_t[0,0]-W_t[1,0])/(W_t[0,2]-W_t[1,2])
    return a * x1 + b

if __name__ == "__main__":
    cls1 = []
    cls2 = []
    
    mean1 = [-1, 2]  
    mean2 = [1, -1]  
    cov = [[1.0,0.8], [0.8,1.0]]

    cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/2))

    temp = vstack((cls1, cls2))
    temp2 = ones((N, 1))  
    X = hstack((temp2, temp))
    
    T = []
    for i in range(N/2):
        T.append(array([1, 0]))  # class1
    for i in range(N/2):
        T.append(array([0, 1]))  # class2
    T = array(T)
    
    X_t = np.transpose(X)
    temp = np.linalg.inv(np.dot(X_t, X))  
    W = np.dot(np.dot(temp, X_t), T)
    W_t = np.transpose(W)
    print W_t
    
    x1, x2 = np.transpose(np.array(cls1))
    plot(x1, x2, 'rx')
    
    x1, x2 = np.transpose(np.array(cls2))
    plot(x1, x2, 'bo')
    
    x1 = np.linspace(-4, 8, 1000)
    x2 = [f(x, W_t) for x in x1]
    plot(x1, x2, 'g-')
    
    xlim(-4, 8)
    ylim(-8, 4)
    show()
