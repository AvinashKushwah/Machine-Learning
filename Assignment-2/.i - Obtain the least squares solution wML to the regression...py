#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

data = open('A2Q2Data_train.csv')
dataset = []
for line in data:
    dataset.append(list(map(float ,line.replace("\n","").split(','))))
dataset = np.array(dataset)
# plt.figure(1)
# plt.scatter(dataset[:,0],dataset[:,1])
yLabel= dataset[:, -1]
dataset= np.delete(dataset, 100, axis=1)
#print(dataset.shape)
#print(yLabel.shape)

temp1 = np.linalg.inv(np.dot(dataset.T,dataset))
temp2 = np.dot(temp1,dataset.T)
temp3 = np.dot(temp2,yLabel)
print(temp3)

