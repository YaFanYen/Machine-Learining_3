# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:19:22 2019

@author: milk
"""
import numpy as np
import matplotlib.pyplot as plt

#1
x=np.linspace(0,1,100)
y=7*x+1.2+0.6*np.random.randn(100)
w=np.zeors(len(x))
b=np.zeros(len(x))
repeat=100
for i in range(repeat):
    hypo=w*x+b
    loss=y-hypo
    error=0.5*(loss**2)/len(x)

#2
def y(x):
    return 2*x-2
x=3
print(y(x))

#3
x=np.linspace(0,1,100)
y=7*x+1.2+0.6*np.random.randn(100)
plt.scatter(x,y)
w=np.zeros(100)
b=np.zeros(100)
lr=0.9
repeat=100
for i in range(repeat):
    hypo=w*x+b
    loss=hypo-y
    error=0.5*sum(loss**2)/len(x)
    grad_w=sum(loss*x)/len(x)
    grad_b=sum(loss)/len(x)
    w=w-lr*grad_w
    b=b-lr*grad_b
    print('iteration:%d | error:%f'%(i,error))
plt.plot(x,hypo,'r')
w2=np.zeros(100)
w1=np.zeros(100)
b0=np.zeros(100)
lr=0.9
repeat=100
plt.scatter(x,y)
for i in range(repeat):
    hypo2=w2*x*x+w1*x+b0
    loss2=hypo2-y
    error2=0.5*sum(loss2**2)/len(x)
    grad_w2=sum(loss2*(x**2))/len(x)
    grad_w1=sum(loss2*x)/len(x)
    grad_b0=sum(loss2)/len(x)
    w2=w2-lr*grad_w2
    w1=w1-lr*grad_w1
    b0=b0-lr*grad_b0
    print('iteration:%d | error:%f'%(i,error2))
plt.plot(x,hypo2,'g')