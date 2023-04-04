#!/usr/bin/env python
# coding: utf-8

# 2.a | Fix a random initialization. For K = {2,3,4,5} , obtain cluster centers according to K-means algorithm using the fixed initialization. For each value of K, plot the Voronoi regions associated to each cluster center. (You can assume the minimum
# and maximum value in the data-set to be the range for each component of R2).

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


def randomCenteroid(dataset,numberOfClusters) :
    min_of_dataset = np.min(dataset)
    max_of_dataset = np.max(dataset)
    kPoints = np.zeros((numberOfClusters, dataset.shape[1]))
    for i in range(numberOfClusters):
        xDim = random.uniform(min_of_dataset, max_of_dataset)
        yDim = random.uniform(min_of_dataset, max_of_dataset)
        kPoints[i][0] = xDim
        kPoints[i][1] = yDim
        
    return kPoints

numberOfClusters= int(input('Number of Clusters:'))
kCentroids=randomCenteroid(dataset,numberOfClusters)
# print(kCentroids)

rows,columns= dataset.shape

# Calulatiing euclidian distance: 

def distanceFromMeans(vector, means):
#     if means.ndim == 1:
#         _sum = np.sum((vector - means)**2)
#     else:
    _sum = np.sum((vector - means)**2, axis = 1)
    distance = np.sqrt(_sum)
    return distance


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
meanColor = ['r', 'g', 'b', 'y','m','c']
for i in range(rows):
    x = dataset[i][0]
    y = dataset[i][1]
    plt.scatter(x,y, color = meanColor[int(Id[i])])

plt.title('cluster formed by picking k random centroid using fix Initialization\n')
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
    if np.array_equal(oldId, Id): break

# Function for finding k random Centroid Points:



# picking k ramdom centroid points : 




for i in range(rows):
    x = dataset[i][0]
    y = dataset[i][1]
    plt.scatter(x,y, color = meanColor[int(Id[i])])
for i in range(numberOfClusters):
    plt.scatter(kCentroids[i][0], kCentroids[i][1], marker='^', color="k", label = "Centroids")
plt.xlabel(' x - axis ')
plt.ylabel('y - axis')
plt.title('Clusters after k-mean converge with fix random intitalization:\n')
plt.show()


#plotting voronoi region associated with each cluster centre: 


from scipy.spatial import Voronoi,voronoi_plot_2d
voronoi_= Voronoi(kCentroids)
vorVertices= voronoi_.vertices
#print(vor_vertices)

vorRegion = voronoi_.regions
#print(vor_region)
fig=voronoi_plot_2d(voronoi_)
plt.show
#print(vornoi_points)
#note : this vornonoi function can only plot voronoi region when clusters are more than 2.


# In[ ]:




