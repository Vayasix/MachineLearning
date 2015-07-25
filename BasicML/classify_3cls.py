#encoding:utf-8
#fig4.5 3cls
import numpy as np
import matplotlib.pyplot as plt

K = 3 #3cls
N = 150

def f1(x1, W_t):
    a = - ((W_t[0,1]-W_t[1,1]) / (W_t[0,2]-W_t[1,2]))
    b = - ((W_t[0,0]-W_t[1,0]) / (W_t[0,2]-W_t[1,2]))
    return a * x1 + b

def f2(x1, W_t):
    a = - ((W_t[1,1]-W_t[2,1]) / (W_t[1,2]-W_t[2,2]))
    b = - ((W_t[1,0]-W_t[2,0]) / (W_t[1,2]-W_t[2,2]))
    return a * x1 + b


if __name__ == "__main__":

    #訓練データ生成
    cls1 = []
    cls2 = []
    cls3 = []

    mean1 = [-2, 2]
    mean2 = [0, 0]
    mean3 = [2, -2]
    cov = [[1.0, 0.8], [0.8, 1.0]]

    #データ生成
    cls1.extend(np.random.multivariate_normal(mean1, cov, N/3))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/3))
    cls3.extend(np.random.multivariate_normal(mean3, cov, N/3))

    #データ行列Xを生成
    temp = np.vstack((cls1, cls2, cls3))
    temp2 = np.ones((N,1))
    X = np.hstack((temp2,temp))

    #ラベル行列作成
    T = []
    for i in range(N/3):
        T.append(np.array([1,0,0]))
    for i in range(N/3):
        T.append(np.array([0,1,0]))
    for i in range(N/3):
        T.append(np.array([0,0,1]))
    T = np.array(T)

    #パラメータ行列Wを最小二乗法で計算
    X_t = np.transpose(X)
    temp = np.linalg.inv(np.dot(X_t, X))
    W = np.dot(np.dot(temp, X_t), T)
    W_t = np.transpose(W)
    print W_t

    #訓練データを描画
    x1, x2 = np.transpose(np.array(cls1))
    plt.plot(x1, x2, 'rx')

    x1, x2 = np.transpose(np.array(cls2))
    plt.plot(x1, x2, 'g+')

    x1, x2 = np.transpose(np.array(cls3))
    plt.plot(x1, x2, 'bo')

    #識別境界２つ　cls1-cls2, cls2-cls3
    x1 = np.linspace(-6, 6, 1000)
    x2 = [f1(x, W_t) for x in x1]
    plt.plot(x1,x2, 'r-')

    x2 = [f2(x, W_t) for x in x1]
    plt.plot(x1,x2, 'b-')

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()


