#!/usr/bin/env python
# coding: utf-8

# Assume that the same data was infact generated from a mixture of Gaussians
# with 4 mixtures. Implement the EM algorithm and plot the log-likelihood (aver-
# aged over 100 random initializations of the parameters) as a function of iterations.
# How does the plot compare with the plot from part (i)? Provide insights that
# you draw from this experiment.

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from numpy.linalg import norm
import random


data = open('A2Q1.csv')
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
# y= dataset[:, -1]
# x= np.delete(dataset, 100, axis=1)
rows,col = dataset.shape

no_mixture = 4


def gaussianDistribution():
    
    #caclulating mean
    mean = np.zeros(col)
    for i in range(col):
        for j in range(rows):
            mean[i] += dataset[j][i]
        mean[i] = mean[i]/rows
        
    
    #calculating covariance
    sigma = np.dot(dataset.T,dataset)
    #print(mean.shape)
    #print(sigma.shape)
    temp =(dataset - mean).T
    
# print(temp)
#     det = numpy.linalg.det(sigma)
#     temp1 = 1 / ((pow((2*3.14),(col/2))) *(math.sqrt(det))) 
#     temp2 =  math.exp(-(1/2)*((dataset - mean).T * (np.linalg.inv(sigma)) * (dataset- mean))
#     print(temp1*temp2)
#     #print(gauss)
    temp1 = ((2 * np.pi) ** (col / 2) * np.linalg.det(sigma) ** (1/2))
    temp2 = np.exp(-0.5 * np.dot(np.dot(temp.T, np.linalg.inv(sigma)), temp))
    return np.diagonal(1 / temp1* temp2)
            
        
        
    
gauss = gaussianDistribution()
#print(gauss.shape)
        

#initialization of GMM and EXPECTATION-MAXIMIZATION:

def Initialization():
    mixtures = []
    
    


# In[ ]:




