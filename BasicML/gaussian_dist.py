import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math


def predict(data):
  mean = np.mean(data)
  var = np.var(data, ddof=1)
  return mean, var



def main():
  #set mean, var
  mean = 3.0
  var = 2.0
  Num_data = 10000
#produce data from gaussian distribution randomly
  data = np.random.normal(mean, var, Num_data)
  mean_pred, var_pred = predict(data)
  standard_pred = math.sqrt(var_pred)
  print ("original: mean={0}, val={1}".format(mean, var))
  print ("predicted : mean={0}, val={1}".format(mean_pred, var_pred))
  
#bins:the number of bar devided, normed: normalization(the integral of the histogram will sum to 1.), alpha:param 
  plt.hist(data, bins=50, normed=True, alpha=0.5)
  xlist = np.linspace(min(data), max(data), 200)
  norm = mlab.normpdf(xlist, mean_pred, standard_pred)
  plt.plot(xlist, norm, 'r-')
  plt.xlim(min(xlist), max(xlist))
  plt.xlabel('x')
  plt.ylabel('probability')
  plt.title('gaussian distribution')
  plt.show()
if __name__ == "__main__":
  main()
