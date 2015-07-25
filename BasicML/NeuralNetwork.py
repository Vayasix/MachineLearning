class NeuralNetwork:
    def __init__(self, NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT, h1, h2):
        #NUM_INPUT: 入力ユニット数（バイアス項を含めない）
        #NUM_HIDDEN: 隠れ層の数（バイアス項を含めない）
        #NUM_OUTPUT: 出力ユニット数
        #h1: 隠れ層の活性化関数
        #h2: 出力層の活性化関数

        self.NUM_INPUT = NUM_INPUT + 1
        self.NUM_HIDDEN = NUM_HIDDEN + 1
        self.NUM_OUTPUT = NUM_OUTPUT
        self.h1 = h1
        self.diff_h1 = diff(h1, x)
        self.h2 = h2
        self.w1 = np.random.random((NUM_INPUT, NUM_HIDDEN))
        self.w2 = np.random.random((NUM_HIDDEN, NUM_OUTPUT))
        self.xi = np.zeros(NUM_INPUT)
        self.zj = np.zeros(NUM_HIDDEN)
        self.zj[0] = 1
        self.yk = np.zeros(NUM_OUTPUT)
        self.dj = np.zeros(NUM_HIDDEN)
        self.dk = np.zeros(NUM_OUTPUT)
    
    #ここを修正しないと実装できない
    def diff(h, x):
        if h == sigmoid:
            return 1 - x*x
        else:
            pass


    def feedForward(self, x):
        # x[0]にbias-term: 1
        self.xi = np.array(x)
        self.xi = np.insert(x, 0, 1)
        # feedForwardProcess
        #first step: calc output of hidden layer
        for j in range(1,self.NUM_HIDDEN):
            self.zj[j] = self.h1(np.dot(np.transpose(w1)[j], self.xi))
        #second step: calc output of output layer
        for k in range(self.NUM_OUTPUT):
            self.yk[k] = self.h2(np.dot(np.transpose(w2)[k], self.zj))
        return self.yk, self.zj
    
    def backPropagation(self, t):
        # back propagation
        self.t = np.array(t)
        self.dk = self.yk - self.t
        for j in range(self.NUM_HIDDEN):
            self.dj[j] = self.diff_h1(self.zj[j]) * np.dot(np.transpose(w2)[j], self.dk)
        return self.dj, self.dk
    
    # 1/2 En(w)
    def En(self):
        return 0.5 * np.dot(self.dk, self.dk)
    
    #　▼En
    def gradEn(self):
        dE_1 = [[self.dj[j]*self.xi[i] for i in range(self.NUM_INPUT)] for j in range(self.NUM_HIDDEN)]
        dE_2 = [[self.dk[k]*self.zj[j] for j in range(self.NUM_HIDDEN)] for k in range(self.NUM_OUTPUT)]
        dE = []
        dE_ = dE_1 + dE_2
        for s in dE_:
            dE.extend(s)
        return dE

    def getW(self):
        temp = []
        for i in self.w1:
            temp.extend(i)
        for j in self.w2:
            temp.extend(j)
        return tuple(temp)

    def setW(self, w):
        w = list(w)
        w_1 = w[0:self.NUM_INPUT*self.NUM_HIDDEN]
        w_2 = w[self.NUM_INPUT*self.NUM_HIDDEN:]
        self.w1 = np.array(w_1).reshape(NUM_INPUT, NUM_HIDDEN)
        self.w2 = np.array(w_2).reshape(NUM_HIDDEN, NUM_OUTPUT)

