#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


M=3
def y(x, wlist):
    ret = wlist[0]
    for i in range(1, M+1):
        ret += wlist[i] * (x ** i)
    return ret


# パラメータWを推定
def estimate(xlist, tlist):
    A = []
    for i in range(M+1):
        for j in range(M+1):
            temp = (xlist**(i+j)).sum()
            A.append(temp)
    A = np.array(A).reshape(M+1, M+1)

    T = []
    for i in range(M+1):
        T.append(((xlist**i) * tlist).sum())
    T = np.array(T)

    wlist = np.linalg.solve(A, T)
    return wlist



def main():
    #訓練データ数
    N = 10
    # sin(2*pi*x)にガウシアンノイズを載せる
    xlist = np.linspace(0, 1, N)
    tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)

    xs = np.linspace(0, 1, 1000)
    ideal = np.sin(2 * np.pi * xs)

    #訓練データからパラメータW_mlを推定
    w_ml = estimate(xlist, tlist)
    print w_ml

    #精度パラメータを推定
    sums = 0
    for n in range(N):
        sums += (y(xlist[n], w_ml) - tlist[n])**2
    beta_inv = 1.0 / N * sums

    means = []
    uppers = []
    lowers = []
    for x in xs:
        m = y(x, w_ml)
        s = np.sqrt(beta_inv)
        u = m + s
        l = m - s
        means.append(m)
        uppers.append(u)
        lowers.append(l)

    plt.plot(xlist, tlist, 'bo')
    plt.plot(xs, ideal, 'g-')
    plt.plot(xs, means, 'r-')
    plt.plot(xs, uppers, 'r--')
    plt.plot(xs, lowers, 'r--')
    plt.xlim(0.0, 1.0)
    plt.ylim(-1.5, 1.5)
    plt.show()




if __name__ == "__main__":
    main()
