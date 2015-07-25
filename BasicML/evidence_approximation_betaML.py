#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

N = 10
alpha = 5 * 10e-3
beta = 11.1


def phi(x,j):
    return x ** j


if __name__ == "__main__":
    #data
    xlist = np.linspace(0,1,N)
    tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0,0.2,N)
    '''
    data = np.matrix([
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
    print data
    '''
    ln_p_list = []
    beta_ml_list = []
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
        beta_ml = N / (np.dot(res, res))
        E_mn = beta_ml/2 * np.dot(res, res) + alpha/2 * np.dot(np.transpose(m_n), m_n)
        ln_p_t = M/2 * np.log(alpha) + N/2 * np.log(beta_ml) - E_mn - 1.0/2 * np.log(np.linalg.norm(A)) - N/2 * np.log(2*np.pi)
        ln_p_list.append(ln_p_t)
        beta_ml_list.append(beta_ml)

    plt.plot(range(10), ln_p_list, 'bo')
    plt.ylim(-25, -5)
    plt.show()
