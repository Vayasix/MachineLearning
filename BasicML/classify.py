#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

K = 2
N = 100 #num of data

def f(x, W_t):
    a = - ((W_t[0,1]-W_t[1,1]) / (W_t[0,2]-W_t[1,2]))
    b = - (W_t[0,0]-W_t[1,0])/(W_t[0,2]-W_t[1,2])
    return a * x + b

def main():
    #訓練データ作成
    cls1 = []
    cls2 = []

    #データを正規分布に従って生成
    mean1 = [-1, 2] # ave of cls1
    mean2 = [1, -1] # ave of cls2
    mean3 = [8,-6]
    cov = [[1.0, 0.8], [0.8, 1.0]] # 共分散行列 (全クラス共通)

    #ノイズなしデータ
    cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/2))
    cls2.extend(np.random.multivariate_normal(mean3, cov, 20))


    #データ行列X生成
    temp = np.vstack((cls1,cls2))
    temp2 = np.ones((N+20,1))
    X = np.hstack((temp2, temp))

    #データ行列T生成
    T = []
    for i in range(N/2):
        T.append(np.array([1,0])) #cls1
    for i in range(N/2+20):
        T.append(np.array([0,1])) #cls2
    T = np.array(T)

    #パラメータ行列Wを最小二乗法で計算
    X_t = np.transpose(X)
    temp = np.linalg.inv(np.dot(X_t, X))
    W = np.dot(np.dot(temp,X_t), T)
    W_t = np.transpose(W)
    print W_t

    #訓練データを描画
    x1, x2 = np.transpose(np.array(cls1))
    plt.plot(x1, x2, 'rx')

    x1, x2 = np.transpose(np.array(cls2))
    plt.plot(x1, x2, 'bo')

    #識別境界を描画
    x1 = np.linspace(-4, 8, 1000)
    x2 = [f(x, W_t) for x in x1]
    plt.plot(x1, x2, 'g-')

    plt.xlim(-4, 8)
    plt.ylim(-8, 4)
    plt.show()




if __name__ == "__main__":
    main()
