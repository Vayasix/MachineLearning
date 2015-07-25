import numpy as np
import matplotlib.pyplot as plt


def gauss():
    m = 0.0
    sigma = 0.3
    xs = np.linspace(-1, 1, 1000)
    reg = np.sqrt(1 / (2 * np.pi * sigma * sigma))
    #ideal = [reg * np.exp(-((x-m)**2)/(2*sigma*sigma)) for x in xs]
    ideal = reg * np.exp(-((xs-m)**2)/(2*sigma*sigma))

    plt.plot(xs, ideal, 'r-')
    plt.xlim(-2.0,2.0)
    plt.ylim(0, 2.0)
    plt.show()

if __name__ == "__main__":
    gauss()
    
