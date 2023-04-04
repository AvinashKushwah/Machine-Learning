#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

dim=float(input('Input value of sigma:'))
k_matrix = kernelMatrixForPolynomial(dataset,dim)
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
index_sort= np.argsort(evalues)[::-1]
sorted_evalues=evalues[index_sort]
sorted_evectors = evectors[:,index_sort]
sorted_evectors=sorted_evectors.T

e1=sorted_evalues[0]
e2=sorted_evalues[1]
ev1=sorted_evectors[0]
ev2=sorted_evectors[1]
N1 = ev1/np.sqrt(e1)
N2=ev2/np.sqrt(e2)
Dim_=np.matrix([N1,N2])
matrix_v = np.dot(centered_kernel_matrix,Dim_.T)
#print(matrix_v)


figure1=pd.DataFrame(matrix_v,columns=['Dimension-1','Dimension-2'])
figure1.plot(kind ='scatter', x='Dimension-1',y='Dimension-2', color ='green')
plt.title('scatter plot')
plt.grid()
plt.show()


# In[ ]:




