#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

N = 4
M = 9
mu = np.linspace(0, 1, M)
s = 0.1
s_sq = s**2

alpha = 2
beta = 25
#gauss_base_func
def phi_gauss(x, j):
    return np.exp(- ((x - mu[j])**2) /(2 * s_sq))


def phi(x, j):
    if j == 0:
        return 1
    else:
        return phi_gauss(x, j-1)

def f(x, w):
    return np.dot(x, w)



if __name__ == "__main__":
    xlist = np.linspace(0,1, N)
    tlist = np.sin(2*np.pi*xlist) + np.random.normal(0, 0.2, N)
    
    xs = np.linspace(0, 1, 1000)
    ideal = np.sin(2*np.pi*xs)
    

    #計画行列Φ
    Phi = np.zeros((N, M+1))
    for n in range(N):
        for j in range(M+1):
            Phi[n, j] = phi(xlist[n],j)
    
    Phi_t = np.transpose(Phi)
    
    Phi_dag = np.dot(np.linalg.inv(alpha * np.identity(M+1) + beta * np.dot(Phi_t, Phi)), Phi_t)
    #平均の重み
    W_ml = beta * np.dot(Phi_dag, tlist)
    print W_ml
    
    def s(x):
        phi_x = np.zeros((M+1))
        for j in range(M+1):
            phi_x[j] = phi(x, j)
        phi_x_t = np.transpose(phi_x)
        S = np.linalg.inv(alpha * np.identity(M+1) + beta * np.dot(Phi_t, Phi))
        s_sqr = 1/beta + np.dot(phi_x_t, np.dot(S, phi_x))
        return np.sqrt(s_sqr)
    
    #出力関数y
    Phy_model = np.zeros((xs.size, M+1))
    for n in range(xs.size):
        for j in range(M+1):
            Phy_model[n, j] = phi(xs[n],j)

    y = np.zeros(xs.size)
    for n in range(len(xs)):
        y[n] = f(Phy_model[n,:], W_ml)
    

    plt.plot(xlist, tlist, 'bo')
    plt.plot(xs, ideal, 'r-')
    
    plt.plot(xs, [y[i] + s(xs[i]) for i in range(len(xs))], 'g-')
    plt.plot(xs, [y[i] - s(xs[i]) for i in range(len(xs))], 'g-')
    
    plt.show()


