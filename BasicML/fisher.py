#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt


N = 100

def f(x, a, b):
    return a * x + b

if __name__ == "__main__":
    #create train data
    cls1 = []
    cls2 = []

    mean1 = [1, 3]
    mean2 = [3, 1]
    cov = [[2.0, 0.0], [0.0, 0.1]]

    cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/2))

    #各クラスの平均をプロット
    m1 = np.mean(cls1, axis=0)
    m2 = np.mean(cls2, axis=0)
    plt.plot(m1[0], m1[1], 'b+')
    plt.plot(m2[0], m2[1], 'r+')
    print m1, m2


    #そうクラス内分散行列を計算
    Sw = np.zeros((2,2))
    for n in range(len(cls1)):
        xn = np.array(cls1[n]).reshape(2,1)
        m1 = np.array(m1).reshape(2,1)
        Sw += (xn-m1) * np.transpose(xn-m1)
    for n in range(len(cls2)):
        xn = np.array(cls2[n]).reshape(2,1)
        m2 = np.array(m2).reshape(2,1)
        Sw += (xn-m2)* np.transpose(xn-m2)
    Sw_inv = np.linalg.inv(Sw)
    w = Sw_inv * (m2-m1)

    #訓練データを描画
    x1, x2 = np.transpose(np.array(cls1))
    plt.plot(x1, x2, 'bo')

    x1, x2 = np.transpose(np.array(cls2))
    plt.plot(x1, x2, 'ro')


    #識別境界を描画
    a = - (w[0,0]/w[1,0])
    m = (m1+m2)/2
    b = - a * m[0,0] + m[1, 0] 

    x1 = np.linspace(-2, 6, 1000)
    x2 = [f(x, a, b) for x in x1]
    plt.plot(x1, x2, 'g-')

    
    plt.xlim(-2, 6)
    plt.ylim(-2, 4)
    plt.show()

    
