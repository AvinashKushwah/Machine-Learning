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

#     from scipy.linalg import eigh
    #centered_k_matrix = centering(k_matrix,r)
    #print(centered_k_matrix)
    row,coloumn=centered_kernel_matrix.shape
    #print(row,coloumn)
    evalues, evectors =eigh(centered_kernel_matrix)
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
    #     onehotencode = np.ones((row, row))/row
    #     partA = np.dot(onehotencode, kernelMat)
    #     partB = np.dot(kernelMat, onehotencode)
    #     partC = np.dot(partA, partB)
    #     res = kernelMat - partA - partB + partC
    #     return res
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


x=input('Type of kernel function(type =polynomial or gaussian):')
numberOfClusters=int(input("Number of Componenets/clusters:"))
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

     
plt.title('')
plt.xlabel('Principal component-1')
plt.ylabel('Principal component-2')
meanColor = ['r', 'g', 'b', 'y','m','c']
zMatrix = np.zeros(old.shape[0])
for i in range(old.shape[0]):
    meanID = np.argmax(dataset[i])
    zMatrix[i] = meanID
    plt.scatter(old[i][0], old[i][1], color = meanColor[int(zMatrix[i])])
plt.show()









# In[ ]:




