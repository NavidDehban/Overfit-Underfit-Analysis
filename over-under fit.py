import numpy as np
import matplotlib.pyplot as plt
from numpy import random

def white_noise():
    mean = 0
    std = 1 
    num_samples = 100
    samples = np.random.normal(mean, std, size=num_samples)
    return samples

def poisson_noise():
    x = random.poisson(lam=2, size=100)
    return x

def estimation(n,x,y):
    a = np.ones((100, n+1))
    for i in range(100):
        for j in range(1,n+1):
            a[i,j] = x[i]**j
    b = np.dot(np.transpose(a),a)
    b = np.linalg.inv(b)
    b = np.dot(b,np.transpose(a))
    b = np.dot(b,y)
    y_predict = np.dot(a,b)
    y_pred.append(y_predict)
    e.append(error(y_predict,y))

def error(y1,y2):
    s = 0 
    for i in range(len(y1)):
        s += (y1[i]-y2[i])**2
    s = s / 100
    return s 

def l_to_m(l):
    res = []
    for i in range(0, len(l)):
        res.append([l[i]])
    res = np.array(res)
    return res

def find_best_n(x,y):
    for n in range(1,16):
        estimation(n,x,y) 
    m = min(e)
    for i in range(len(e)):
        if e[i] == m:
            print('best n:',i+1)

def print_list(e):
    for i in range(len(e)):
        print('error n = '+str(i+1)+':',e[i])
    
def plot_estimation(x,y_pred):
    plt.plot(x,y_pred[0])
    plt.plot(x,y_pred[3])
    plt.plot(x,y_pred[6])
    plt.plot(x,y_pred[14])
    plt.legend(['n = 1','n = 4','n = 7','n =15'])
    plt.show()

def avg(l):
    return sum(l)/len(l)
 
def variance(y_pred):
    v = []
    for y_predict in y_pred:
        a = avg(y_predict)
        for i in range(len(y_predict)):
            y_predict[i] = (y_predict[i] - a)**2
        v.append(avg(y_predict))
    return v

def bias(y1,y_pred):
    b = []
    v = variance(y_pred)
    for j in range(len(y_pred)):
        y2 = y_pred[j]
        e = []
        for i in range(len(y1)):
            e.append((y1[i]-y2[i])**2)  
        e = avg(e)
        b.append((e - v[j])**0.5)
    for i in range(len(b)):
        print('bias for n =',i+1,b[i])

def print_variance(v):
    for i in range(len(v)):
        print('variance for n =',i+1,v[i])

def main():
    x = np.arange(-10 , 10 , 0.2)
    y = 2 * np.cos(x)/-np.pi + 2 * np.sin(2 * x)/(2 * np.pi) + 2 * np.cos(3 * x)/(-3 * np.pi)
    y = 0.12*(poisson_noise()+white_noise()) + y
    plt.plot(x,y)
    plt.figure()
    y = l_to_m(y)
    x = l_to_m(x)
    find_best_n(x,y)
    print(".......................")
    print_list(e)
    plot_estimation(x,y_pred)
    print(".......................")
    v = variance(y_pred)
    print_variance(v)
    print(".......................")
    bias(y,y_pred)
e = []
y_pred = []
main()





