#coding:utf-8

# 非線形SVM
# cvxoptのQuadratic Programmingを解く関数を使用

import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers
from pylab import *


N = 100
P = 3 # 多項式カーネルのパラメータ
SIGMA = 5.0 #ガウスカーネルのパラメーター

def polynomial_kernel(x,y):
    return (1 + np.dot(x, y)) ** P

def gaussian_kernel(x,y):
    return np.exp(- (norm(x-y)**2)/ (2 * (SIGMA**2)))

#どちらかのカーネルを選択
kernel = gaussian_kernel

def f(x, a, t, X, b):
    sum = 0.0
    for n in range(N):
        sum += a[n] * t[n] * kernel(x, X[n])
    return sum + b


if __name__ == "__main__":
    #訓練データ生成
    cls1 = []
    cls2 = []
    
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    
    cls1.extend(np.random.multivariate_normal(mean1, cov, N/4))
    cls1.extend(np.random.multivariate_normal(mean3, cov, N/4))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/4))
    cls2.extend(np.random.multivariate_normal(mean4, cov, N/4))

    X = vstack((cls1, cls2))

    t = []
    for i in range(N/2):
        t.append(1.0)   # クラス1
    for i in range(N/2):
        t.append(-1.0)  # クラス2
    t = array(t)
    
    # ラグランジュ乗数を二次計画法（Quadratic Programming）で求める
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = t[i] * t[j] * kernel(X[i], X[j])
    
    Q = cvxopt.matrix(K)
    p = cvxopt.matrix(-np.ones(N))
    G = cvxopt.matrix(np.diag([-1.0]*N))
    h = cvxopt.matrix(np.zeros(N))
    A = cvxopt.matrix(t, (1,N))
    b = cvxopt.matrix(0.0)
    sol = cvxopt.solvers.qp(Q, p, G, h, A, b)
    a = array(sol['x']).reshape(N)
    print a

    S = []
    for i in range(len(a)):
        if a[i] < 1e-5:
            continue
        S.append(i)

    sum = 0.0
    for n in S:
        temp = 0.0
        for m in S:
            temp += a[m] * t[m] * kernel(X[n], X[m])
        sum += (t[n] - temp)
    b = sum / len(S)

    print S, b
    
    # 訓練データを描画
    x1, x2 = np.array(cls1).transpose()
    plot(x1, x2, 'rx')
    
    x1, x2 = np.array(cls2).transpose()
    plot(x1, x2, 'bx')

    for n in S:
        scatter(X[n, 0], X[n, 1], s = 80, c = 'g', marker='o')
        
    
    """ここらへんの書き方復習"""
    #識別境界を描画
    X1, X2 = meshgrid(linspace(-6,6,50), linspace(-6,6,50))
    w, h = X1.shape
    X1.resize(X1.size)
    X2.resize(X2.size)
    Z = array([f(array([x1, x2]), a, t, X, b) for (x1, x2) in zip(X1, X2)])
    
    X1.resize((w, h))
    X2.resize((w, h))
    Z.resize((w, h))
    CS = contour(X1,  Z)
    #CS = contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    """--------------------------"""
    
    
    for n in S:
        print f(X[n], a, t, X, b)
    
    xlim(-6, 6)
    ylim(-6, 6)
    show()

