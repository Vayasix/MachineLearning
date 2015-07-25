#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

"""
勾配降下法で線形回帰による曲線フィッティング
解析的に解ける正規方程式ではなく、逐次的に解く
"""
def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []
    for iter in range(iterations):
        temp = np.dot(X, theta) - y
        theta = theta - alpha * (1.0/m) * np.dot(X.T, temp)
        J_history.append(computeCost(X, y, theta))
    return theta, J_history


def computeCost(X, y, theta):
    m = len(y)
    temp = np.dot(X, theta) - y
    J = (1.0/(2*m)) * np.dot(temp, temp)
    return J

def plotData(X, y):
    plt.scatter(X, y, c='red', marker='o', label="Training data")
    plt.xlabel('x')
    plt.ylabel('y')

if __name__ == "__main__":
    #訓練データを生成
    X = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, X.size)

    #訓練データ数
    m = len(y)

    #訓練データをプロット
    plt.figure(1)
    plotData(X, y)

    #訓練データの１列目に１を追加
    X = X.reshape((m, 1))
    X = np.hstack((np.ones((m,1)), X, X**2, X**3))

    #パラメーターを０で初期化
    theta = np.zeros(4)
    iterations = 100000
    alpha = 0.2

    #初期状態のコストを計算
    initialCost = computeCost(X, y, theta)
    print 'initialCost: ', initialCost

    #勾配降下法でパラメーターを推定
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    print 'theta:', theta
    print 'final cost:', J_history[-1]

    #コストの履歴をプロット
    plt.figure(2)
    plt.plot(J_history)
    plt.xlabel("iterations")
    plt.ylabel("J(theta)")

    #曲線をプロット
    plt.figure(1)
    plt.plot(X[:, 1], np.dot(X, theta), 'b-', label='Linear regression')
    plt.legend()
    plt.xlim(0,1)
    plt.show()

