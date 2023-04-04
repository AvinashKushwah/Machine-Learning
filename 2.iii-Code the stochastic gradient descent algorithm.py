#!/usr/bin/env python
# coding: utf-8

# Code the stochastic gradient descent algorithm using batch size of 100 and plot || w - wml || as a function of t. What are your observations?

# In[1]:


#attempt 4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from numpy.linalg import norm
import random


data = open('A2Q2Data_train.csv')
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
y= dataset[:, -1]
x= np.delete(dataset, 100, axis=1)



#calculation wml = (xTx)-1 xTy

r1= np.linalg.inv(np.dot(x.T,x))
r2= np.dot(r1,x.T)
wml= np.dot(r2,y)



#Gradient(w) = 2((xTx)w - xTy)

def Gradient(w,X,Y):
    z1= np.dot(X.T,X)
    temp1 = np.dot(z1,w)
    return 2*(temp1 - np.dot(X.T,Y))

# Func(w) = || w-wml || : L2-Norm

def Func(w):
#     r1= np.linalg.inv(np.dot(X.T,X))
#     r2= np.dot(r1,X.T)
#     wml= np.dot(r2,Y)
    return np.linalg.norm(w-wml)
#     temp = 0
#     z = np.array([0 for i in range(100)])
#     for i in range(100) :
#         z[i] = wml[i] - w[i]
#         temp = temp + z[i]*z[i]
        
#     return math.sqrt(temp)

      
    

    
    
def gradient_function(X,Y,iteration):
    w = np.array([0 for i in range(100)])
    T= iteration
    StepSize=1
    List = []
    for i in range (iteration):
#         np.random.shuffle(dataset)
#         X= dataset[:100,:100]
#         Y= dataset[:100,100:101]
        sigma = 0.0000001/StepSize
        w = w - sigma *Gradient(w,X,Y)
        StepSize = StepSize +1
    List.append(Func(w))
        
    return List



# In stochatic graident descent, we have to uniformly pick batch(size =100) at random,So there can be two approaches to pick datapoint at random-
# Aproach 1 : we randomly select 100 data points, but it may happen that some data points are used more than once for k iteration
# Approach 2: we shuffle rows of dataset, and pick data first 100 datapoint in first iteration, then 100-199 in second iteration and so on.
# I will be using Approach 1 for my stochastic gradient descent computation.

def StochasticGradientDescent(iteration):
    i=1;
    ans=[]
    while(i<20):
        np.random.shuffle(dataset)
        X= dataset[:100,:100]
        Y= dataset[:100,100:101]
        ans.append(gradient_function(X,Y,iteration))
        i = i+1
    return ans



iteration =500
ans = StochasticGradientDescent(iteration)




plt.xlabel('Iteration>')
plt.ylabel('Least square error>')
plt.title('||W-W_ml|| Vs Iteration')
plt.plot( ans)
plt.show()
    


# In[ ]:




