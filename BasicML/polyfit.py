import numpy as np
import matplotlib.pyplot as plt


N = 10

# M_dim poly approximation
def y(x, wlist, M):
    ret = wlist[0]
    for i in range(1, M+1):
        ret += wlist[i] * (x ** i)
    return ret

# estimate param from train data
def estimate(xlist, tlist, M):
    
    # A_ij = sum(x_n ^(i+j))
    A = []
    for i in range(M+1):
        for j in range(M+1):
            temp = (xlist**(i+j)).sum()
            A.append(temp)
    A = np.array(A).reshape(M+1,M+1)

    # T_i = sum(x_n ^i * t_n)
    T = []
    for i in range(M+1):
        T.append(((xlist**i) * tlist).sum())
    T = np.array(T)

    #param: w 
    wlist = np.linalg.solve(A, T)

    return wlist

def estimate_reg(xlist, tlist, lam, M):
    # A_ij = sum(x_n ^(i+j))
    A = []
    for i in range(M+1):
        for j in range(M+1):
            temp = (xlist**(i+j)).sum()
            if i == j:
                temp += lam
            A.append(temp)
    A = np.array(A).reshape(M+1,M+1)

    # T_i = sum(x_n ^i * t_n)
    T = []
    for i in range(M+1):
        T.append(((xlist**i) * tlist).sum())
    T = np.array(T)

    #param: w 
    wlist = np.linalg.solve(A, T)

    return wlist


def plot_function_name(name, x=0, y=0):
    plt.text(x,y,name, alpha=0.3, size=20, ha="center", va="center")

def subfig(xlist, tlist, M, xs, ideal, n, name):
    
    # select 
    #wlist = estimate(xlist, tlist, M)
    wlist = estimate_reg(xlist, tlist, np.exp(-18.0),M)
    
    model = [y(x, wlist, M) for x in xs]

    plt.subplot(2, 2, n)
    plt.plot(xlist, tlist, 'bo')
    plt.plot(xs, ideal, 'g-')
    plt.plot(xs, model, 'r-')
    plot_function_name(name)
    plt.xlim(0.0,1.0)
    plt.ylim(-1.5,1.5)


def polyfit():
    
    #produce train data:N
    xlist = np.linspace(0, 1, N)
    tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)

    
    #original sin data
    xs = np.linspace(0,1,1000)
    ideal = np.sin(2 * np.pi * xs)
    #model = [y(x, wlist, M) for x in xs]

    
    subfig(xlist, tlist, 0, xs, ideal, 1, "M=0")
    subfig(xlist, tlist, 1, xs, ideal, 2, "M=1")
    subfig(xlist, tlist, 3, xs, ideal, 3, "M=3")
    subfig(xlist, tlist, 9, xs, ideal, 4, "M=9")
    plt.show()
    
if __name__ == "__main__":
    polyfit()

