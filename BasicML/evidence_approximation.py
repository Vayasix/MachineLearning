#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

N = 10
alpha = 5 * 10e-3
beta = 11.1
s = 0.1
s_sq = s**2
#mu = np.linspace(0, 1, 10)
# ガウス基底関数
def phi_gauss(x, j):
    return np.exp(-1*((x - mu[j])**2)/(2* s_sq))
    
# 多項式基底関数定義

def phi(x,j):
    return x ** j


if __name__ == "__main__":
    ###data
    #xlist = np.linspace(0,1,N)
    #tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0,0.2,N)
    ###tlist = np.sin(2 * np.pi * xlist)
    data = np.array([
        [0.000000, 0.349486],
        [0.111111, 0.830839],
        [0.222222, 1.007332],
        [0.333333, 0.971507],
        [0.444444, 0.133066],
        [0.555556, 0.166823],
        [0.666667, -0.848307],
        [0.777778, -0.445686],
        [0.888889, -0.563567],
        [1.000000, 0.261502],
        ])
    
    xlist = data[:,0]
    tlist = data[:,1]
    
    ln_p_list = []

    for M in range(10):
        Phi = np.zeros((N, M+1))
        for n in range(N):
            for j in range(M+1):
                Phi[n,j] = phi(xlist[n], j)
        Phi_t = np.transpose(Phi)
        #define m_n, A
        A = alpha * np.identity(M+1) + beta * np.dot(Phi_t, Phi)
        A_inv = np.linalg.inv(A)
        m_n = beta * np.dot(A_inv, np.dot(Phi_t, tlist))
        #ln(Evidence)
        res = tlist - np.dot(Phi, m_n)
        E_mn = beta/2.0 * np.dot(res, res) + alpha/2.0 * np.dot(np.transpose(m_n), m_n)
        #ここで det(A)ではなく、norm(A)を使うことに注意
        #ln_p_t = M/2 * np.log(alpha) + N/2 * np.log(beta) - E_mn - 1.0/2 * np.log(np.linalg.det(A)) - N/2 * np.log(2*np.pi)
        ln_p_t = M/2.0 * np.log(alpha) + N/2.0 * np.log(beta) - E_mn - 1.0/2.0 * np.log(np.linalg.norm(A)) - N/2.0 * np.log(2*np.pi)
        ln_p_list.append(ln_p_t)
    
    print ln_p_list
    plt.plot(range(10), ln_p_list, 'bo')
    plt.ylim(-25, -10)
    plt.show()
