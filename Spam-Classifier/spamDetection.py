#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from numpy.linalg import eigh
import pandas as pd
import string
from numpy import mean,std,cov
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import os
import re
from striprtf.striprtf import rtf_to_text



dataset= pd.read_csv('spam.csv',encoding = 'latin-1')





# spam.csv consist of five columns where column no 3,4,5 are useless.So dropping these columns.
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)        
#print(dataset.shape)
dataset.rename(columns = {'v1': 'Label', 'v2': 'message'}, inplace = True)    
#dataset.describe()


# calculating the percentage of message labeled spam/ham.
dataset['Label'].value_counts(normalize=True)     


# changing labels: ham --> 0, spam--> 1
dataset.loc[dataset['Label'] == 'spam','Label',] = 1          # spam = 1
dataset.loc[dataset['Label'] == 'ham','Label',] =  0          # ham = 0



# data preprocessing:
def dataCleaning(X):
    X['message'] = X['message'].str.replace('\W', ' ') # Removing punctuation
    X['message'] = X['message'].str.lower()            # converting messages into lower case
    return X

# dataset after clearning
dataset = dataCleaning(dataset)  
#print(dataset)


#Appending some baggy words that occur in spam emails
ListofSpamWords = "Billion compensation dollar prize fake win lottery compensation package winning credit crediting amount  due to losses and Sandy and lottery beneficiaries that originated  Your approved Compensation package has been deposited in the Security Vault waiting for delivery. For identification and swift delivery of your compensation package, you are advice to contand re-confirm your delivery details earn cash extra million free only for you free trial refund  get paid offer insurance make money offer expires"
ListofSpamWords = ListofSpamWords.translate(str.maketrans('', '', string.punctuation))
ListofSpamWords = ListofSpamWords.lower() 
buggyWords =[]
buggyWords= ListofSpamWords.split()
#print(buggyWords)

#splitting dataset for training and testing 
ShuffledData = dataset.sample(frac= 1)      # shuffling dataset uniformly at random
#print(shuffledData)

row,col = ShuffledData.shape
SplitIndex = int(row * 0.8)              # spliting data into 80-20.

trainData = ShuffledData[:SplitIndex]
testData = ShuffledData[SplitIndex:]

#print(trainData.shape)
#print(testData.shape)
trainData['Label'].value_counts(normalize=True)



df = pd.DataFrame(trainData,columns =['Label','message'])
#print(df)

#seprating nonSpamEmails and SpamEmails:
nonSpamData = df.loc[df['Label'] == 0] 
spamData = df.loc[df['Label'] == 1] 


#creating vocabulary for dataset and returning number of words in vocabulary without removing repeted words, list of unique
# words and list of words without removing repeted words.
def createVocabulary(training_set):
    training_set['message'] = training_set['message'].str.split()

    vocabulary = []
    for sms in training_set['message']:
        for word in sms:
            vocabulary.append(word)
    size = len(vocabulary)
    vocabularyWithRepetion = vocabulary
    vocabulary = list(set(vocabulary))
    return size,vocabulary,vocabularyWithRepetion



ntrainData,dtrainData,trainDataR = createVocabulary(trainData) # creating vocabulary for trainData
nTestData,dTestData,testDataR = createVocabulary(testData)     #creating vocabulary for test Data
len1,v1,r1 = createVocabulary(spamData)                        #creating vocabulary for spamData
len2,v2,r2 = createVocabulary(nonSpamData)                     #creating vocabulary for nonSpamData
#print(dtrainData)



for i in buggyWords:
    dtrainData.append(i)

# df = pd.DataFrame(trainData,columns =['Label','message'])
# #print(df)
# nonSpamData = df.loc[df['Label'] == 0] 
# spamData = df.loc[df['Label'] == 1] 


    


# FindNumberOfWords is used to calculate conditional probablity of word given label(First we count number of occurance of word in list of emails with given label and then dividing with total number of words)    
def FindNumberOfWords(label,word):
    count =0;
    temp = r1
    if(label == 1):
        temp =r1
    if(label == 0):
        temp = r2
    for i in range (len(temp)):
        if(word == temp[i]):
            count = count + 1
    
    return count
    
    
    
    
# Calulating Prior Probablities of label 1 and Label 0.
def priorProbablity(spamData,nonSpamData):        #calculating prior
    
    x = len(spamData)
    y = len(nonSpamData)
    priorX = x/(x+y)
    priorY = y/(y+x)
    return priorX,priorY



PriorX,PriorY = priorProbablity(spamData,nonSpamData)




def findConditionalProb(word,label):
    
    wordsInSpam= len1
    wordsInNonSpam = len2
    if(label == 1):
        temp = FindNumberOfWords(label,word)
        return temp/wordsInSpam
    if(label == 0):
        temp = FindNumberOfWords(label,word)
        return temp/wordsInNonSpam
    return 1
        


    
# computing posterior = prior * p(word1/label)*p(word2/label)...

def Computing(label, email):
    temp =1
    for i in range(len(email)):
        temp = temp * findConditionalProb(email[i],label)
    
    if(label == 1):
        temp = temp*PriorX
    if(label == 0):
        temp = temp*PriorY
        
    return temp
    
    
    
    
    
    
    
    
    
# classify : this function is taking email and our traindata and returing list of keyword of email that are present in our train data and ignoring others.    
def classify(email,dtrainData):
    email = email.translate(str.maketrans('', '', string.punctuation))
    email = email.lower()            
    wordsInEmail = []
    temp = []
    wordsInEmail = email.split()
    for i in range(len(wordsInEmail)):
        for j in range(len(dtrainData)):
            if(wordsInEmail[i] == dtrainData[j]):
                temp.append(wordsInEmail[i])
                break;
     
    return temp


#strInput = input('write an email')

#temp = classify(strInput,dtrainData)


#This function is caculating whether p(y=1/email) > p(y=0/email) and vice versa and if p(y=1/email) > pp(y=0/email) return spam else return not spam.
def FindClass(temp):
    p1 = 1
    p0 = 0
    temp1 = Computing(p1,temp)
    temp2 = Computing(p0,temp)
    if(temp1 > temp2):
        return 1
        #print('SPAM')
    else:
        return 0
        #print('NO SPAM')
    
    
    
# testData  = testData.to_numpy()
# print(testData[0][0])



#Finding accuracy of spam detection on test Data.
def Find_Accuracy(testData):
    testData  = testData.to_numpy()
    rows,cols= testData.shape
    accuracy = 0
    for i in range (rows):
        temp = testData[i][1]
        str1 = " "
        str1 = str1.join(temp)
        inputString = classify(str1,dtrainData)
        res = FindClass(inputString)
        if(res == testData[i][0]):
            accuracy = accuracy + 1
    
    percentage = accuracy/rows
    incorrect = rows - accuracy
    percentage = percentage * 100;
    return percentage,incorrect,accuracy


# predictionAccuracy,IncorrectPrediction,accuracy = Find_Accuracy(testData)
# print("Accuracy on test Data:",predictionAccuracy)
# print("Number of Incorrect Prediction:",IncorrectPrediction)
# print("number of correct Prediction:",accuracy)


# Reading the text file present in the folder test.

file = os.listdir('test/')
test_data = []

for email in file:
    with open('test/'+email) as infile:
        content = infile.read()
        text = rtf_to_text(content)
    test_data.append(text)
    

#print(len(test_data))


# Testing On Test Data Provided:

for i in range(len(test_data)):
    
    inputString = classify(test_data[i],dtrainData)
    res= FindClass(inputString)
    if(res == 1):
        print("Email:",test_data[i])
        print("This is a Spam Email")
    if(res == 0):
        print("Email:",test_data[i])
        print("This is Not a Spam Email")
    









        
        

    
    


# In[ ]:





# In[ ]:




