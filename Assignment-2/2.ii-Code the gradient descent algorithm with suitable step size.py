#!/usr/bin/env python
# coding: utf-8

# Code the gradient descent algorithm with suitable step size to solve the least
# squares algorithms and plot || w - wml || as a function of t. What do you
# observe?

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from numpy.linalg import norm


data = open('A2Q2Data_train.csv')
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
# plt.figure(1)
# plt.scatter(dataset[:,0],dataset[:,1])
yLabel= dataset[:, -1]
dataset= np.delete(dataset, 100, axis=1)
#mean = np.mean(dataset,axis=0)
#dataset= dataset-mean
print(dataset.shape)
print(yLabel.shape)


xTx = np.dot(dataset.T,dataset)          # calculation xTx required for wml 
xTy = np.dot(dataset.T,yLabel)           # calculating xTy required for wml



#calculation wml = (xTx)-1 xTy

r1= np.linalg.inv(xTx)
r2= np.dot(r1,dataset.T)
wml= np.dot(r2,yLabel)



#Gradient(w) = 2((xTx)w - xTy)

def Gradient(w):
    temp1 = np.dot(xTx,w)
    return 2*(temp1 - xTy)

# Func(w) = || w-wml || : L2-Norm

def Func(w):
#     temp = 0
#     z = np.array([0 for i in range(100)])
#     for i in range(100) :
#         z[i] = wml[i] - w[i]
#         temp = temp + z[i]*z[i]
        
#     return math.sqrt(temp)
      return np.linalg.norm(w-wml)
    

    
    
def gradient_function(iteration):
    w = np.array([0 for i in range(100)])
    T= iteration
    StepSize=1
    List = []
    for i in range (1000):
        sigma = 0.0000001/StepSize
        w = w - sigma *Gradient(w)
        StepSize = StepSize +1
        List.append(Func(w))
        
    return List
        
no_iteration = 1000
res = gradient_function(no_iteration)
#print(res)


iteration = [ i+ 1 for i in range(1000)]
plt.xlabel('Iteration>')
plt.ylabel('Least square error>')
plt.plot(iteration, res)
plt.show()
    


# In[ ]:




