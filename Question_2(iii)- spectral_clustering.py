#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import eigh
import pandas as pd
from numpy import mean,std,cov
import matplotlib.pyplot as plt
import random
import math

data = open('Dataset.csv')
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
#print(dataset)
plt.figure(1)
plt.scatter(dataset[:,0],dataset[:,1])

#procedure of centering a kernal matrix :

def PolynomialKernel(dataset,d,k):

    def centering(kernelMat,row):

        onehotdecode=np.ones((row,row))
        onehotencode=np.eye(row,row) - onehotdecode/1000
        part1 = np.dot(onehotencode,kernelMat)
        part2 = np.dot(part1,onehotencode)
        return part2

    def kernelMatrixForPolynomial(dataset,d):
        row,column=dataset.shape
        kernelMat = np.zeros((row,row))
        #print(kernelMat)
        for r in range(row):
            for c in range(row):
                kernelMat[r,c]=pol_kernel(dataset[r],dataset[c],d)
        return kernelMat

    def pol_kernel(i,j,d):
        res = np.dot(i,j)
        res = res+ 1
        res = np.power(res,d)
        return res

    # computing Kernel matrix of size N*N 
    k_matrix = kernelMatrixForPolynomial(dataset,d)
    #print(k_matrix)
    r,c = k_matrix.shape
    #print(r,c)
    centered_kernel_matrix = centering(k_matrix,r)
    #print(centered_kernel_matrix)

    from scipy.linalg import eigh
    #centered_k_matrix = centering(k_matrix,r)
    #print(centered_k_matrix)
    row,coloumn=centered_kernel_matrix.shape
    #print(row,coloumn)
    evalues, evectors =np.linalg.eig(centered_kernel_matrix)
    #print(evalues)
    k = numberOfClusters
    index_sort= np.argsort(evalues)[-k:][::-1]
    sorted_evalues=evalues[index_sort]
    sorted_evectors = evectors[:,index_sort]
    sorted_evectors=sorted_evectors.astype('float64')
    return sorted_evectors


def RBFKernel(dataset,d,k):
    #procedure of centering a kernal matrix :

    def centering(kernelMat,row):

        onehotdecode=np.ones((row,row))
        onehotencode=np.eye(row,row) - onehotdecode/1000
        part1 = np.dot(onehotencode,kernelMat)
        part2 = np.dot(part1,onehotencode)
        return part2

    def kernelMatrixForPolynomial(dataset,d):
        row,column=dataset.shape
        kernelMat = np.zeros((row,row))
        #print(kernelMat)
        for r in range(row):
            for c in range(row):
                kernelMat[r,c]=gaussian_kernel(dataset[r],dataset[c],d)
        return kernelMat

    def gaussian_kernel(i,j,sigma):
        return np.exp((-(np.linalg.norm(i-j)**2))/(2*sigma**2))

    # computing Kernel matrix of size N*N :

    
    k_matrix = kernelMatrixForPolynomial(dataset,d)
    #print(k_matrix)
    r,c = k_matrix.shape
    #print(r,c)
    centered_kernel_matrix = centering(k_matrix,r)
    #print(centered_kernel_matrix)
    from scipy.linalg import eigh
    #centered_k_matrix = centering(k_matrix,r)
    #print(centered_k_matrix)
    row,coloumn=centered_kernel_matrix.shape
    #print(row,coloumn)
    evalues, evectors =np.linalg.eig(centered_kernel_matrix)
    #print(evalues)
    index_sort= np.argsort(evalues)[-k:][::-1]
    sorted_evalues=evalues[index_sort]
    sorted_evectors = evectors[:,index_sort]
    sorted_evectors=sorted_evectors.astype('float64')
    
    return sorted_evectors
   





from random import choices

# Storing the dataset.csv file in dataset Array:
data = open('Dataset.csv')
old = []
for line in data:
    old.append(list(map(float ,line.replace("\n","").split(','))))
old = np.array(old)


x=input('Type of kernel function( type =polynomial or gaussian):')
numberOfClusters=int(input("Number of Clusters:"))
if(x =='polynomial'):
    p=int(input('Degree of polynomial:'))
    dataset = PolynomialKernel(dataset,p,numberOfClusters)

else:
    sigma=float(input('input value of sigma:'))
    dataset=RBFKernel(dataset,sigma,numberOfClusters)
     
def normalise(data):
    for i in range(data.shape[0]):
        row_square_sum = np.sum(data[i]**2)
        normalizing_factor = np.sqrt(row_square_sum)
        data[i] = data[i]/normalizing_factor
    return data
dataset = normalise(dataset)






#dataset = sorted_evectors
# Function for finding k random Centroid Points:

def randomCenteroid(dataset,numberOfClusters) :
    kPoints = choices(dataset,k=numberOfClusters)
    #print(kPoints)
    #print(kPoints[0])
    return kPoints


kCentroids=randomCenteroid(dataset,numberOfClusters)
#print(kCentroids


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
meanColor = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
rows = old.shape[0]  
for i in range(rows):
    x = old[i][0]
    y = old[i][1]
    plt.scatter(x,y, color = meanColor[int(Id[i])])
plt.xlabel(' x - axis ')
plt.ylabel('y - axis')
plt.title('clusters after k-mean algorithm converges')
plt.show()


# In[ ]:




