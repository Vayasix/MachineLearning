import numpy as np
import matplotlib.pyplot as plt
from pylab import *



N = 100
ETA = 0.1

if __name__ == "__main__":
    #tarin data
    cls1 = []
    cls2 = []
    t = []

    mean1 = [-2, 2]
    mean2 = [2, -2]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    
    cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/2))

    #target data
    for i in range(N/2):
        t.append(+1)
    for i in range(N/2):
        t.append(-1)

    #discrive train data
    x1, x2 = np.transpose(np.array(cls1))
    plt.plot(x1, x2, 'bo')

    x1, x2 = np.transpose(np.array(cls2))
    plt.plot(x1, x2, 'ro')

    #merge
    x1, x2 = np.array(cls1+cls2).transpose()
    
    #reset param w
    w = np.array([1.0,1.0,1.0])
    
    turn = 0
    correct = 0

    while correct < N:
        correct = 0
        for i in range(N):
            if np.dot(w, [1, x1[i], x2[i]])*t[i] > 0:
                correct += 1
            else:
                w += ETA * np.array([1,x1[i],x2[i]]) * t[i]
        turn += 1
        print turn, w

    #draw boundary
    x = np.linspace(-6.0, 6.0, 1000)
    y = - w[1]/w[2] * x - w[0]/w[2]
    plt.plot(x, y, 'g-')
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.show()
