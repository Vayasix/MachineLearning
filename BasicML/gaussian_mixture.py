#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math


def gaussian(x, mean, cov):
  #multivariate gaussian distribution
  temp1 = 1/((2*np.pi) **(x.size/2.0))
  temp2 = 1/(np.linalg.det(cov)**0.5)
  temp3 = -0.5 * np.dot(np.dot(x-mean, np.linalg.inv(cov)), x-mean)
  return temp1 * temp2 * np.exp(temp3)



def choice_color(gamma):
  temp = np.zeros((len(gamma),1))
  print temp
  color_table = np.hstack((gamma, temp))
  print color_table
  for n in range(N):   
    max_res = max(gamma[n][0], gamma[n][1], gamma[n][2])
    if max_res > 0.8:
      Xcolor[n] = colorset(max_res)

      

def main():
# produce data from gaussian distribution
  # mean for each distribution
  mean1 = [0.25,0.25]
  mean2 = [0.5,0.5]
  mean3 = [0.75,0.75]
  mean = [mean1, mean2, mean3]


  # covariance for each distribution
  cov1 = [[0.02, 0.05], [0.007, 0.01]]
  cov2 = [[0.02, -0.03], [-0.01, 0.01]]
  cov3 = [[0.02, 0.05], [0.007, 0.01]]
  cov = [cov1, cov2, cov3]

  #prodece data
  cls1 = []
  cls2 = []
  cls3 = []
  N=600 # total of data
  K=3 # total of gaussian distribution



  cls1.extend(np.random.multivariate_normal(mean1, cov1, N/3))
  cls2.extend(np.random.multivariate_normal(mean2, cov2, N/3))
  cls3.extend(np.random.multivariate_normal(mean3, cov3, N/3))

  X = np.vstack((cls1, cls2, cls3))
  print X

  # responsibility
  gamma = np.zeros((N, K))
  pi = np.random.rand(K)

  for n in range(N):
    denominator = 0.0
    for j in range(K):
      denominator += pi[j] * gaussian(X[n], mean[j], cov[j])
    # calc responsibility
    for k in range(K):
      gamma[n][k] = pi[k] * gaussian(X[n], mean[k], cov[k]) / denominator 
  print gamma

 # choice_color(gamma)
  

  # calc real mean on each class
  m1 = np.mean(cls1, axis=0)
  m2 = np.mean(cls2, axis=0)
  m3 = np.mean(cls3, axis=0)
  plt.plot(m1[0], m1[1], 'r+')
  plt.plot(m2[0], m2[1], 'g+')
  plt.plot(m3[0], m3[1], 'b+')
  print("mean: cls1={0}, cls2={1}, cls3={2}".format(m1, m2, m3))
  


  # plot class data
  #class1
  x1, x2 = np.array(cls1).transpose()
  plt.plot(x1, x2, 'ro')
  #class2
  x1, x2 = np.array(cls2).transpose()
  plt.plot(x1, x2, 'go')
  #class3
  x1, x2 = np.array(cls3).transpose()
  plt.plot(x1, x2, 'bo')

  plt.xlim(-0.1, 1.1)
  plt.ylim(-0.1, 1.1)

  plt.show()


if __name__ == "__main__":
  main()
