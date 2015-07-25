#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

NUM_DATA = 50
ETA = 0.1
NUM_LOOP = 1000
NUM_INPUT = 2 #includes bias-term
NUM_HIDDEN = 4 #includes bias-term
NUM_OUTPUT = 1


def sum_of_squares_error(xlist, tlist, w1, w2):
    error = 0.0
    for n in range(NUM_DATA):
        z = np.zeros(NUM_HIDDEN)
        y = np.zeros(NUM_OUTPUT)
        #insert 1 to the head
        x = np.insert(xlist[n], 0, 1)
        for j in range(NUM_HIDDEN):
            a = np.zeros(NUM_HIDDEN)
            for i in range(NUM_INPUT):
                a[j] += w1[j, i] * x[i]
            z[j] = np.tanh(a[j])
        for k in range(NUM_OUTPUT):
            for j in range(NUM_HIDDEN):
                y[k] += w2[k, j] * z[j]
        # calc ２乗和誤差
        for k in range(NUM_OUTPUT):
            error += 0.5 * (y[k]-tlist[n,k])*(y[k]-tlist[n,k])
    return error


def output(x, w1, w2):
    # 配列に変換して先頭にバイアスの1を挿入
    x = np.insert(x, 0, 1)
    z = np.zeros(NUM_HIDDEN)
    y = np.zeros(NUM_OUTPUT)
    # 順伝播で出力を計算
    for j in range(NUM_HIDDEN):
        a = np.zeros(NUM_HIDDEN)
        for i in range(NUM_INPUT):
            a[j] += w1[j, i] * x[i]
        z[j] = np.tanh(a[j])
    for k in range(NUM_OUTPUT):
        for j in range(NUM_HIDDEN):
            y[k] += w2[k, j] * z[j]
    return y, z



if __name__ == "__main__":
    xlist = np.linspace(0, 1, NUM_DATA).reshape(NUM_DATA,1)
    tlist = (np.sin(2*np.pi*xlist.reshape(1, NUM_DATA)) + np.random.normal(0, 0.2, NUM_DATA)).reshape(NUM_DATA,1)
    #tlist = np.sin(2*np.pi*xlist)
    w1 = np.random.random((NUM_HIDDEN, NUM_INPUT))
    w2 = np.random.random((NUM_OUTPUT, NUM_HIDDEN))
    
    print "学習前の二乗誤差：", sum_of_squares_error(xlist, tlist, w1, w2)
    
    for loop in range(NUM_LOOP):
        #update the weights according to n
        for n in range(NUM_DATA):
            z = np.zeros(NUM_HIDDEN)
            y = np.zeros(NUM_OUTPUT)

            d1 = np.zeros(NUM_HIDDEN)
            d2 = np.zeros(NUM_OUTPUT)

            x = np.array([xlist[n]])
            x = np.insert(x, 0, 1)

            for j in range(NUM_HIDDEN):
                a = np.zeros(NUM_HIDDEN)
                for i in range(NUM_INPUT):
                    a[j] += w1[j, i] * x[i]
                z[j] = np.tanh(a[j])

            for k in range(NUM_OUTPUT):
                for j in range(NUM_HIDDEN):
                    y[k] += w2[k, j] * z[j]

            for k in range(NUM_OUTPUT):
                d2[k] = y[k] - tlist[n, k]

            for j in range(NUM_HIDDEN):
                temp = 0.0
                for k in range(NUM_OUTPUT):
                    temp += w2[k,j] * d2[k]
                d1[j] = (1-z[j]*z[j]) * temp

            #update
            for j in range(NUM_HIDDEN):
                for i in range(NUM_INPUT):
                    w1[j, i] -= ETA * d1[j] * x[i]

            for k in range(NUM_OUTPUT):
                for j in range(NUM_HIDDEN):
                    w2[k, j] -= ETA * d2[k] * z[j]

        print loop, sum_of_squares_error(xlist, tlist, w1, w2)

    #学習後の重みを更新
    print "w1:", w1
    print "w2:", w2
    
    #テスト　訓練データ入力＆隠れ、出力ユニットの出力を計算
    ylist = np.zeros((NUM_DATA, NUM_OUTPUT))
    zlist = np.zeros((NUM_DATA, NUM_HIDDEN))
    for n in range(NUM_DATA):
        ylist[n], zlist[n] = output(xlist[n], w1, w2)
    
    plt.plot(xlist, tlist, 'bo')
    plt.plot(xlist, ylist, 'r-')

    #for i in range(NUM_HIDDEN):
     #   plt.plot(xlist, zlist[:,i], 'k--')
    
    plt.ylim(-1.5,1.5)
    plt.show()
    






