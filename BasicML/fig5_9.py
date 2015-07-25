import numpy as np
import matplotlib.pyplot as plt
import pylab
#num of train data
N = 10
#the learning rate
ETA = 0.1
#num of loops
NUM_LOOP = 1000
# num of units in input layer (including bias term)
NUM_INPUT = 2
# num of units in hidden layer (including bias term)
NUM_HIDDEN = 10
# num of units in output layer 
NUM_OUTPUT = 1

def output(x, w1, w2):
    x = np.insert(x, 0, 1)
    z = np.zeros(NUM_HIDDEN)
    y = np.zeros(NUM_OUTPUT)
    
    for j in range(NUM_HIDDEN):
        a = np.zeros(NUM_HIDDEN)
        for i in range(NUM_INPUT):
            a[j] += np.transpose(w1)[j, i] * x[i]
        z[j] = np.tanh(a[j])

    #second step: calc output of output layer
    for k in range(NUM_OUTPUT):
        for j in range(NUM_HIDDEN):
            y[k] += np.transpose(w2)[k,j] * z[j]
    
    return y, z


if __name__ == "__main__":
    
    xlist = np.linspace(0, 1, N).reshape(N,1)
    tlist = (np.sin(2*np.pi*xlist.reshape(1,N)) + np.random.normal(0.0, 0.2, xlist.size)).reshape(N,1)

    #ideal model sin
    xs = np.linspace(0,1, 1000)
    ideal = np.sin(2*np.pi*xs)

    #vector of weight w1, w2
    w1 = np.random.random((NUM_INPUT, NUM_HIDDEN))
    w2 = np.random.random((NUM_HIDDEN, NUM_OUTPUT))

    #loop until converge or until E <=1 espilon
    for loop in range(NUM_LOOP):
        for n in range(len(xlist)):
            z = np.zeros(NUM_HIDDEN)
            y = np.zeros(NUM_OUTPUT)

            d1 = np.zeros(NUM_HIDDEN)
            d2 = np.zeros(NUM_OUTPUT)

            x = np.array(xlist[n])
            x = np.insert(x, 0, 1)

            #first step: calc output of hidden layer
            for j in range(NUM_HIDDEN):
                a = np.zeros(NUM_HIDDEN)
                for i in range(NUM_INPUT):
                    a[j] += np.transpose(w1)[j, i] * x[i]
                z[j] = np.tanh(a[j])

            #second step: calc output of output layer
            for k in range(NUM_OUTPUT):
                for j in range(NUM_HIDDEN):
                    y[k] += np.transpose(w2)[k,j] * z[j]

            #evaluate E
            for k in range(NUM_OUTPUT):
                d2[k] = y[k] - tlist[n,k]

            # back propagation
            for j in range(NUM_HIDDEN):
                temp = 0.0
                for k in range(NUM_OUTPUT):
                    temp += np.transpose(w2)[k,j] * d2[k]
                d1[j] = (1 - z[j]*z[j]) * temp
            
            # update w1
            for j in range(NUM_HIDDEN):
                for i in range(NUM_INPUT):
                    np.transpose(w1)[j, i] -= ETA * d1[j] * x[i]

            for k in range(NUM_OUTPUT):
                for j in range(NUM_HIDDEN):
                    np.transpose(w2)[k, j] -= ETA * d2[k] * z[i]
    
    ylist = np.zeros((N, NUM_OUTPUT))
    zlist = np.zeros((N, NUM_HIDDEN))

    for n in range(N):
        ylist[n], zlist[n] = output(xlist[n], w1, w2)


    print tlist
    plt.plot(xlist, tlist, 'bo')
    plt.plot(xlist, ylist, 'r-')
    plt.xlim(-0.1,1.1)
    plt.ylim(-1.5,1.5)
    plt.show()
        


