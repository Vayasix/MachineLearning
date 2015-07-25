#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

N = 10
M = 9
lam = 0.000001

def phi(x, j):
    return x**j

def y(x, w):
    return np.dot(x,w)

if __name__ == "__main__":
    xlist = np.linspace(0, 1, N)
    tlist = np.sin(2*np.pi*xlist) + np.random.normal(0.0, 0.2, xlist.size)

    xs = np.linspace(0,1,1000)
    ideal = np.sin(2*np.pi*xs)
    
    

    #計画行列Φ
    Phi = np.zeros((N, M+1))
    for n in range(N):
        for j in range(M+1):
            Phi[n, j] = phi(xlist[n],j)
    
    #ムーア・ベンローズの擬似逆行列
    Phi_t = np.transpose(Phi)
    
    Phi_dag = np.dot(np.linalg.inv(lam * np.identity(M+1) + np.dot(Phi_t,Phi)), Phi_t)
    #平均の重み
    W_ml = np.dot(Phi_dag, tlist)
    print W_ml
    
    #出力関数y
    Phy_model = np.zeros((xs.size, M+1))
    for n in range(xs.size):
        for j in range(M+1):
            Phy_model[n, j] = phi(xs[n],j)

    model = np.zeros(xs.size)
    for n in range(len(xs)):
        model[n] = y(Phy_model[n,:], W_ml)

    

    #ここからプロット
    plt.plot(xs, ideal, 'g-')
    plt.plot(xlist, tlist, 'bo')
    plt.plot(xs, model, 'r-')
    plt.xlim(-0.001,1.001)
    plt.ylim(-1.5, 1.5)
    plt.show()

