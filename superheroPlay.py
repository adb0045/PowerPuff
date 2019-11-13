#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Mutual information and SVC section
'''
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
#import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# User defined methods
#returns the dataset that has be reduced by finding the mutual information and choosing the features 
#that are greater than the given threshold
def getMutualInfo(data, target, threshold):
    #Mutual information (MI) [1] between two random variables is a non-negative value, 
    #which measures the dependency between the variables. It is equal to zero if and only if 
    #two random variables are independent, and higher values mean higher dependency.
    mi = mutual_info_classif(data, target, random_state = 10)
    highVals = {}
    keys = []
    for index,val in enumerate(mi):
        if val > threshold:
            highVals[index] = val
            keys.append(index)
    #print(highVals)
    dataMI = data.iloc[:,keys]
    return dataMI


def getTestTrainSet(data, target):
    X_train, X_test, y_train, y_test = [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]

    # get all the training and testing data
    # setting the random_state so that I can get consistent results while testing 
    j = 0.1 #training set percentage
    for i in range(5):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(data,target,train_size=j, random_state=100)
        j += 0.1
    
    return X_train, X_test, y_train, y_test

# input parameter are each an array of 5 datasets corresponding to their namesake
# returns the set of corresponding predictions
def getSetOfPreds(Xtrain, Xtest,ytrain, ytest):
    svc_score = []

    for i in range(5):
        clf = SVC(kernel='linear', gamma='auto')
        clf.fit(Xtrain[i], ytrain[i])
        svc_y_pred = clf.predict(Xtest[i])
        svc_score.append((clf.score(Xtrain[i], ytrain[i]))*100)
        #print(confusion_matrix(ytest[i], svc_y_pred))
    return svc_score


# In[3]:


# read in the datafile
df1=pd.read_csv('heroes_information.csv')
df2=pd.read_csv('super_hero_powers.csv')

#df1.info()
#df1.head()


# In[4]:


print ('Number of dimensions:',df1.ndim)
print ('Shape of dataset1:',df1.shape)
df1.isnull().sum()
#find number of duplicate names
print('Number duplicate names',df1['name'].duplicated().sum())


# In[5]:


df2.info()
print ('\n Number of dimensions:',df2.ndim)
print ('Shape of dataset1:',df2.shape)
df2.isnull().sum()
print('Number duplicate names',df2['hero_names'].duplicated().sum())


# In[6]:


#df2.head()


# In[7]:


#counts for hero alignment, '-' denotes unknown
df1['Alignment'].value_counts()


# In[8]:


#rename column in df2 for merge
df2.rename(columns={'hero_names':'name'},inplace=True)
#df2.head()


# In[9]:


#combine datasets with name as index
df_new=df1.set_index('name').join(df2.set_index('name'))
df_new.to_csv('df_new.csv')
#print(df_new)


# In[10]:


#df_new.isnull().sum()
#df_new.head()


# In[11]:


#1 if x=="good" else 0 if x=="neutral" or "Unknown" else -1 if x=="bad" else x
f=lambda x:1 if x==True or x=='good' else 0 if x==False or x=='neutral' or x=='-'else -1 if x=="bad" else x

adjusted=df_new.applymap(f)


# In[12]:


a=adjusted.drop(columns=['Gender','Unnamed: 0','Eye color','Hair color','Race','Publisher','Height','Skin color','Weight'])


# In[13]:


a.shape


# In[14]:


# exclude samples with missing feature information
a=a.dropna()

# set y equal to the alignment
y=a.iloc[:,0]

# set XOrig to all features except the alignment
XOrig=a.iloc[:,1:]
print(XOrig.shape)

# set up the other datasets using mutual information as the feature selection mechanism
Xsmall = getMutualInfo(XOrig, y, 0.03)
print(Xsmall.shape)
Xmed = getMutualInfo(XOrig, y, 0.01)
print(Xmed.shape)
Xlarge = getMutualInfo(XOrig, y, 0.001)
print(Xlarge.shape)


# In[15]:


#get the test train set for every dataset
X_train, X_test, y_train, y_test = getTestTrainSet(XOrig, y)
Xs_train, Xs_test, ys_train, ys_test = getTestTrainSet(Xsmall, y)
Xm_train, Xm_test, ym_train, ym_test = getTestTrainSet(Xmed, y)
Xl_train, Xl_test, yl_train, yl_test = getTestTrainSet(Xlarge, y)


# In[16]:


# now do svc on whole data set before and after and get the accuracy results
X_scores = getSetOfPreds(X_train, X_test, y_train, y_test)
Xs_scores = getSetOfPreds(Xs_train, Xs_test, ys_train, ys_test)
Xm_scores = getSetOfPreds(Xm_train, Xm_test, ym_train, ym_test)
Xl_scores = getSetOfPreds(Xl_train, Xl_test, yl_train, yl_test)


# In[17]:


print(Xs_scores)
print()

print(Xm_scores)
print()

print(Xl_scores)
print()
print(X_scores)


# In[18]:


#plot classification accuracy vs training size (training size is x-axis and accuracy is y axis)

# x-axis values in %
train_split = [10,20,30,40,50]

plt.figure(figsize=(10,7))
plt.plot(train_split, X_scores, 'ro-', label = 'SVC - Full set')
plt.plot(train_split, Xl_scores, 'yo-', label = 'SVC - large set')
plt.plot(train_split, Xm_scores, 'go-', label = 'SVC - medium set')
plt.plot(train_split, Xs_scores, 'bo-', label = 'SVC - small set')


plt.ylabel('Accuracy %')
plt.xlabel('Training Set Size (%)')
plt.legend()
plt.show()

