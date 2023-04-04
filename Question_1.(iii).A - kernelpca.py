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

# computing Kernel matrix of size N*N :
dim=int(input("Input degree of polynomial :\n"))
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
#print(sorted_evalues)
#print(sorted_evectors[0])
#evalues, evectors = evalues[::-1], evectors[:, ::-1]
#print(evalues)

#transforming data :
# evalue1=sorted_evalues[0]
# evalue2=sorted_evalues[1]
# evector1=sorted_evectors[0]
# evector2=sorted_evectors[1]
# Normalized_evector1= evector1/np.sqrt(evalue1)
# Normalized_evector2=evector2/np.sqrt(evalue2)
# #print(Normalized_evector1)
# Eigen_vector_matrix = np.matrix([Normalized_evector1,Normalized_evector2])
# # result=np.zeros((row,dim),dtype='complex_')
# # #print(Eigen_vector_matrix)
# # result= np.matmul(centered_kernel_matrix.T,Eigen_vector_matrix.T)
# # print(result)
e1=sorted_evalues[0]    
e2=sorted_evalues[1]
#print(e1,e2)
ev1=sorted_evectors[0]
ev2=sorted_evectors[1]
#print(ev1)
#print(ev2)
N1 = ev1/np.sqrt(e1)  # Normalized eigen vector corresponding to maximum eigen value
N2=ev2/np.sqrt(e2)    #Normalized eigen vector corresponding to 2nd maximum eigen value.
# alpha = np.zeros((2, row))
# for i in range(2):
#     alpha[i] = sorted_evectors[i] / np.sqrt(sorted_evalues[i])
#print(alpha)
Dim_=np.matrix([N1,N2])
#print(Dim_)
matrix_v = np.dot(centered_kernel_matrix,Dim_.T)
#print(matrix_v)

figure1=pd.DataFrame(matrix_v,columns=['Dimension-1','Dimension-2'])
figure1.plot(kind ='scatter', x='Dimension-1',y='Dimension-2', color ='green')
plt.title('scatter plot')
plt.grid()
plt.show()


# In[ ]:




