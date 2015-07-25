#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

M = 9
ALPHA = 0.005
#ALPHA = 2.0
BETA = 11.1
#BETA = 25

def y(x, wlist):
    ret = wlist[0]
    for n in range(1, M+1):
        ret += wlist[i] * (x ** i)
    return ret

def phi(x):
    data = []
    for i in range(0, M+1):
        data.append(x**i)
    ret = np.array(data).reshape(M+1, 1)
    return ret

def mean(x, xlist, tlist, S):
    sums = np.matrix(np.zeros((M+1, 1)))
    for n in range(len(xlist)):
        sums += phi(xlist[n]) * tlist[n]
    ret = BETA * phi(x).transpose() * S * sums
    return ret

def variance(x, S):
    ret = 1.0 / BETA + phi(x).transpose() * S * phi(x)
    return ret


def main():
    # data Num: N
    N = 4
    # train data: {x, t}
    xlist = np.linspace(0, 1, N)
    xlist[1] += 0.1
    xlist[2] -= 0.1
    tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)

    xs = np.linspace(0, 1, 1000)
    ideal = np.sin(2*np.pi*xs)

    #ベイズ曲線フィッティングを用いて、予測分布を求める
    #行列Sを計算
    sums = np.matrix(np.zeros((M+1, M+1)))
    for n in range(len(xlist)):
        sums += phi(xlist[n]) * phi(xlist[n]).transpose()
    I = np.matrix(np.identity(M+1))
    S_inv = ALPHA * I + BETA * sums
    S = S_inv.getI()


    means = []
    uppers = []
    lowers = []
    for x in xs:
        m = mean(x, xlist, tlist, S)[0,0]
        s = np.sqrt(variance(x, S)[0,0])
        u = m+s
        l = m-s
        means.append(m)
        uppers.append(u)
        lowers.append(l)
        
    plt.plot(xlist, tlist, 'bo')
    plt.plot(xs, ideal, 'g-')
    plt.plot(xs, means, 'r-')
    plt.plot(xs, uppers, 'r--')
    plt.plot(xs, lowers, 'r--')
    plt.xlim(0, 1)
    plt.ylim(-1.5, 1.5)
    plt.show()


if __name__ == "__main__":
    main()
