#!/usr/bin/env python
# coding: utf-8

# # Clustering 
# Run the K-means algorithm with K = 4 on the same data. Plot the objective of
# K - means as a function of iterations.

# In[12]:


import numpy as np
from numpy.linalg import eigh
import pandas as pd
from numpy import mean,std,cov
import matplotlib.pyplot as plt
import random
from random import choices
# Storing the dataset.csv file in dataset Array:

data = open('A2Q1.csv')               
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
print(dataset.shape)


# Function for finding k random Centroid Points:

def randomCenteroid(dataset,numberOfClusters) :
    kPoints = choices(dataset,k=numberOfClusters)
    #print(kPoints)
    #print(kPoints[0])
    return kPoints

# picking k ramdom centroid points : 

numberOfClusters= int(input('Number of Clusters:'))      
kCentroids=randomCenteroid(dataset,numberOfClusters)


rows,columns= dataset.shape

# Calulatiing euclidian distance: 

def distanceFromMeans(vector, means):
#     if means.ndim == 1:
#         _sum = np.sum((vector - means)**2)
#     else:
    _sum = np.sum((vector - means)**2, axis = 1)
    distance = np.sqrt(_sum)
    return distance

errorFunction = []
Id = np.zeros(rows)
min_dist = np.zeros(rows)
for i in range(rows):
    #temp = np.zeros(4)
    dMeans = distanceFromMeans(dataset[i], kCentroids)
    meanId = np.argmin(dMeans)
    min_dist[i] = dMeans[meanId]
    Id[i]=meanId
    #dMeans.idxmin(axis=1)
#     print(dMeans )
#print(Id)
errorFunction.append(np.sum(min_dist))
meanColor = ['r', 'g', 'b', 'y','m','c']

# k-mean algorithm :

while True :
    oldId= Id.copy()
    #print(kCentroids)
    for i in range(numberOfClusters):
        kCentroids[i] = np.zeros(columns)
        n = 0
        for j in range(rows):
            if Id[j] == i:
                kCentroids[i] += dataset[j]
                n += 1
        kCentroids[i] /= n
    for i in range(rows):
        temp1 = distanceFromMeans(dataset[i],kCentroids)
        meanId1= np.argmin(temp1)
        if temp1[meanId1] < min_dist[i]:
            Id[i]=meanId1
            min_dist[i] = temp1[meanId1]
    errorFunction.append(np.sum(min_dist))
    if np.array_equal(oldId, Id): break


iteration = [ i+1 for i in range(len(errorFunction))]
plt.xlabel('Iteration ->')
plt.ylabel('Error Function ->')
plt.plot(iteration, errorFunction)
plt.show()


# In[ ]:




