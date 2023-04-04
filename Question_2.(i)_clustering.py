#!/usr/bin/env python
# coding: utf-8

# # Clustering 
# Write a piece of code to run the algorithm studied in class for the K-means
# problem with k = 4 . Try 5 different random initialization and plot the error
# function w.r.t iterations in each case. In each case, plot the clusters obtained in
# different colors.

# In[1]:


import numpy as np
from numpy.linalg import eigh
import pandas as pd
from numpy import mean,std,cov
import matplotlib.pyplot as plt
import random
from random import choices
# Storing the dataset.csv file in dataset Array:

data = open('Dataset.csv')
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
#print(dataset)
plt.figure(1)
plt.scatter(dataset[:,0],dataset[:,1])

# Function for finding k random Centroid Points:

def randomCenteroid(dataset,numberOfClusters) :
    kPoints = choices(dataset,k=numberOfClusters)
    #print(kPoints)
    #print(kPoints[0])
    return kPoints

# picking k ramdom centroid points : 

numberOfClusters= int(input('Number of Clusters:'))
kCentroids=randomCenteroid(dataset,numberOfClusters)
#print(kCentroids)

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
for i in range(rows):
    x = dataset[i][0]
    y = dataset[i][1]
    plt.scatter(x,y, color = meanColor[int(Id[i])])


plt.show()

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


for i in range(rows):
    x = dataset[i][0]
    y = dataset[i][1]
    plt.scatter(x,y, color = meanColor[int(Id[i])])
    
    
for i in range(numberOfClusters):
    plt.scatter(kCentroids[i][0], kCentroids[i][1], marker='^', color="k", label = "Centroids")
plt.xlabel(' x - axis ')
plt.ylabel('y - axis')
plt.title('clusters after K-mean converges:')
plt.show()
iteration = [ i+ 1 for i in range(len(errorFunction))]
plt.xlabel('Iteration ->')
plt.ylabel('Error Function ->')
plt.plot(iteration, errorFunction)
plt.show()


# In[ ]:




