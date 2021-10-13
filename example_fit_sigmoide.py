""" Example sigmoid fit
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import itertools

def sigmoid(x, a, b, c):
    y = a / (1 + np.exp(-b * (x - a))) + c
    return y

a = 80 # value with x=0
b = 0.1 # speed of decay
c = 10 # value in infinity
x =  np.linspace(1, 150, 100)
y = sigmoid(x, a, b, c)

plt.plot(x, y)
plt.show()


def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

a = 80 # value with x=0
b = 0.1 # speed of decay
c = 10 # value in infinity
x =  np.linspace(0, 100, 100)
y = exponential(x, a, b, c)

plt.plot(x, y)
plt.show()

def logaritmic(x, a, b, c):
    import numpy as np
    return a * np.log(b * x) + c


a = 80 # value with x=0
b = 0.1 # speed of decay
c = 10 # value in infinity
x =  np.linspace(1, 100, 100)
y = logaritmic(x, a, b, c)

plt.plot(x, y)
plt.show()




""" exploratory with sigmoid
"""
a_ = np.arange(1, 10000, 1000)# [100, 0.95e+03, 1e+03, 1.05e+03, 1.1e+03]
b_ = np.arange(0, 0.01, 0.001)# [9e-03, 8e-03, 7e-03, 6e-03, 5e-03]
x =  np.linspace(0, 2000, 100)
y_hat = []
combinations = []
for a, b in itertools.product(a_, b_):
    combinations.append('a:{}-b:{}'.format(a,round(b,4)))
    y_hat.append(sigmoid(x, a, b))


plt.figure()
for i in np.arange(0, len(y_hat)-1):
    plt.plot(x, y_hat[i], 'o', label=combinations[i])
#plt.legend(loc='best')
plt.show()


""" exploratory with exponential
"""
a_ = np.arange(1, 120, 10)# [100, 0.95e+03, 1e+03, 1.05e+03, 1.1e+03]
b_ = np.arange(-0.01, 0.01, 0.005)# [9e-03, 8e-03, 7e-03, 6e-03, 5e-03]
c_ = np.arange(1, 50, 10)# [9e-03, 8e-03, 7e-03, 6e-03, 5e-03]

y_hat = []
combinations = []
for a, b, c in itertools.product(a_, b_, c_):
    combinations.append('a:{}-b:{}'.format(a,round(b,4)))
    y_hat.append(exponential(x, a, b, c))


plt.figure()
for i in np.arange(0, len(y_hat)-1):
    plt.plot(x, y_hat[i], 'o', label=combinations[i])
#plt.legend(loc='best')
plt.show()





if 1 == 0:

    plt.plot(x, y_hat[1], 'o', label='data', color='blue')
    plt.plot(x, y_hat[2], 'o', label='data', color='black')
    plt.plot(x, y_hat[3], 'o', label='data', color='pink')
    #plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.show()

    xdata = np.array([400, 600, 800, 1000, 1200, 1400, 1600])
    ydata = np.array([0, 0, 0.13, 0.35, 0.75, 0.89, 0.91])
    x = np.linspace(-1, 2000, 50)

    popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox', p0=[1000, 0.001])
    print(popt)
    y_hat1 = sigmoid(x, *popt)

    plt.figure()
    plt.plot(xdata, ydata, 'o', label='data')
    plt.plot(x, y_hat1, label='fit1')
    #plt.plot(x, y_hat2, label='fit2')
    #plt.plot(x, y_hat3, label='fit3')
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.show()




a = 2.5
b = 1.3
c = 0.5
yfunc = exponential(x, a, b, c)

popt, pcov = curve_fit(exponential, x, y)
yfunc = func(xs, *popt)