import numpy as np
import matplotlib.pyplot as plt

np.random.seed()
N = 100000
Nlin = 20
x = np.random.uniform(0.0, 1.0, N)
xs = np.linspace(0.0, 1.0, Nlin)
y = np.zeros(Nlin)
for i in range(len(x)):
  for j in range(len(xs)):
    if x[i] > xs[j] and x[i] < xs[j+1]:
      y[j] += 1
print y[:10]
y = y / N

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 0.07)
plt.plot(xs, y, '-')
plt.plot(xs, y, '|')
plt.show()
