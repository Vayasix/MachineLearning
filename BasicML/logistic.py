#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from pylab import *

N = 100

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def f(x,w):
    a = - (w[1]/w[2])
    b = - (w[0]/w[2])
    return a * x + b

if __name__ == "__main__":
    #train data 
    cls1 = []
    cls2 = []
    
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [8, -6]
    cov = [[1.0,0.8], [0.8,1.0]]

    cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/2-20))
    cls2.extend(np.random.multivariate_normal(mean3, cov, 20))

    #データ行列X
    temp = np.vstack((cls1, cls2))
    temp2 = np.ones((N,1))
    X = np.hstack((temp2, temp))

    #label T
    t = []
    for i in range(N/2):
        t.append(1.0)
    for i in range(N/2):
        t.append(0.0)
    t = np.array(t)

    #param w <- re
    turn = 0
    w = array([0.0, 0.0, 0.0])
    while True:
        #phi calc
        phi = X
        #calc R y
        R = np.zeros((N,N))
        y = []
        for n in range(N):
            a = np.dot(w, phi[n,])
            y_n = sigmoid(a)
            R[n,n] = y_n * (1 - y_n)
            y.append(y_n)

        #ヘッセ行列H
        phi_T = np.transpose(phi)
        H = np.dot(phi_T, np.dot(R, phi))

        #w <-w
        w_new = w - np.dot(np.linalg.inv(H), np.dot(phi_T, y-t))

        #w の収束判定
        diff = np.linalg.norm(w_new - w) / np.linalg.norm(w)
        print turn, diff
        if diff < 0.1:
            break

        w = w_new
        turn += 1
    
    #訓練データを描画
    x1, x2 = np.array(cls1).transpose()
    plt.plot(x1, x2, 'rx')
    
    x1, x2 = np.array(cls2).transpose()
    plt.plot(x1, x2, 'bo')
    
    # 識別境界を描画
    x1 = np.linspace(-6, 10, 1000)
    x2 = [f(x, w) for x in x1]
    plt.plot(x1, x2, 'g-')
    
    xlim(-6, 10)
    ylim(-10, 6)
    plt.show()
