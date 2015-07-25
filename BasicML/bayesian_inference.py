#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt


MEAN = 0.8
VAR = 0.1
MEAN0 = 0.0
VAR0 = 0.1

def gauss_distribution(xlist, mean, var):
    normarize = 1/np.sqrt(2*np.pi*var)
    ret = normarize * np.exp(- ((xlist-mean)**2)/(2*var))
    return ret

def calc_mean_n(m0, m_ml, v0, v1, N):
    print m_ml
    ret = (v1/(N*v0 + v1))*m0 + (N*v0/(N*v0+v1))*m_ml
    print ret
    return ret

def calc_var_n(v0, v1, N):
    var_n_inv = 1/v0 + N/v1
    return 1 / var_n_inv

def calc_m_ml(data, N):
    sample = []
    i = 0
    while i != N:
        get = np.random.choice(data, 1)
        if (get > 0):
            sample.append(get)
            i += 1
        else:
            continue
    sample = np.array(sample)
    return sample.sum() / N

def main():
    xs = np.linspace(-1, 1, 1000)
    xlist = np.linspace(0, 2, 1000)
    data = gauss_distribution(xlist, MEAN, VAR)
    
    #N=0
    g0 = gauss_distribution(xs, MEAN0, VAR0)
    
    #N=1
    m1 = calc_mean_n(MEAN0, calc_m_ml(data,1), VAR0, VAR, 1)
    v1 = calc_var_n(VAR0, VAR, 1)
    g1 = gauss_distribution(xs, m1, v1)

    #N=2
    m2 = calc_mean_n(MEAN0, calc_m_ml(data,2), VAR0, VAR, 2)
    v2 = calc_var_n(VAR0, VAR, 2)
    g2 = gauss_distribution(xs, m2, v2)
    
    #N=10
    m10 = calc_mean_n(MEAN0, calc_m_ml(data,10), VAR0, VAR, 10)
    v10 = calc_var_n(VAR0, VAR, 10)
    g10 = gauss_distribution(xs, m10, v10)
    
    m50 = calc_mean_n(MEAN0, calc_m_ml(data,50), VAR0, VAR, 50)
    v50 = calc_var_n(VAR0, VAR, 50)
    g50 = gauss_distribution(xs, m50, v50)

    plt.plot(xs, g0, 'k-')
    plt.plot(xs, g1, 'g-')
    plt.plot(xs, g2, 'b-')
    plt.plot(xs, g10, 'r-')
    plt.plot(xs, g50, 'c-')
    plt.xlim(-1, 1)
    plt.ylim(0, 5)
    plt.title("bayesian inference for N")
    plt.show()

if __name__ == "__main__":
    main()
