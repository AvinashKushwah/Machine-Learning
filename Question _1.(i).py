#!/usr/bin/env python
# coding: utf-8

# # Principal component Analysis:
# Write a piece of code to run the PCA algorithm on this data-set. How much of
# the variance in the data-set is explained by each of the principal components?

# In[7]:


import numpy as np
from numpy.linalg import eigh
import pandas as pd
from numpy import mean,std,cov
import matplotlib.pyplot as plt
import random

# Storing Dataset in a dataset array:

data = open('Dataset.csv')
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
#print(dataset)
plt.xlabel('x-axis')
plt.ylabel('Y-axis')
plt.title('Initial dataset')
plt.figure(1)
plt.scatter(dataset[:,0],dataset[:,1])
plt.show()


#calculate mean

means = np.array([0.0,0.0])
for i in dataset:
    means[0] += i[0]
    means[1] += i[1] 

means = means/1000


# centering dataset matrix :

center = dataset- means
center
plt.figure(2)
plt.xlabel('x-axis')
plt.ylabel('Y-axis')
plt.title('dataset after centering')
plt.scatter(center[:,0], center[:,1], zorder=1)
plt.show()

#caluclation of covariance matrix : 

covariance = np.dot(center.T, center)/1000

#print(covariance)


# Finding Eigen values and eigen vectors


values, vectors =eigh(covariance)
values = values[::-1]
vectors = vectors[::-1]

#print(values, vectors)

sum_of_ev=np.sum(values)


#Showing principal components and their variance Explained :

vectors = vectors.T
center = dataset- means
origin = [0,0]
c = ['r','g','y','m']
i = 0
for elm in vectors.T:
    
    
    plt.figure(i + 2)
    variance_cov = (values[i] / sum_of_ev) *100 
    plt.title(f"Principle component : {i+1} \n\n E.Value = {values[i]} \nE.Vector = {str(elm)}\n  Variance explained = {variance_cov}%\n")
    plt.xlabel('x-axis')
    plt.ylabel('Y-axis')
    w = [elm[0], elm[1]]
    plt.axline(origin, w, color = random.choice(c))
    plt.scatter(center[:,0], center[:,1], zorder=1)
    plt.show()
    i+=1



# In[ ]:




