#encoding:utf-8
from scipy.linalg import *
import numpy as np
#初期設定
alpha = 2.0
beta = 25
mu = np.array([0,0])
sigma = np.matrix([[alpha, 0], [0, alpha]])
a0 = -0.3
a1 = 0.5
xv = [0.9, -0.6]
tv = [0.05, -0.8]
for xn in range(18):
    xn = 2*np.random.random()-1
    xv += [xn]
    tv += [a0+a1*xn+np.random.normal(0, 0.2, )]
