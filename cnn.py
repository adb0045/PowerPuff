#!/bin/python3

"""
Ian Hines
CS488 FA19
PowerPuff
"""

import keras
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit

# Read in data
df1=pd.read_csv('heroes_information.csv')
df2=pd.read_csv('super_hero_powers.csv')

# Merge datasets
df2.rename(columns={'hero_names':'name'},inplace=True)
df_new=df1.set_index('name').join(df2.set_index('name'))
df_new.to_csv('df_new.csv')

# Drop useless columns and convert columns that need to be numbers into numbers
f=lambda x:1 if x==True else 0 if x==False else 2 if x=="good" else 0 if x=="bad" else 1 if x=="neutral" or "Unknown" else x
adjusted=df_new.applymap(f)
a=adjusted.drop(columns=['Gender','Unnamed: 0','Eye color','Hair color','Race','Publisher','Height','Skin color','Weight'])
a=a.dropna()

# Separate dependent and independent variables
X=a.iloc[:,1:]
Y=a.iloc[:,0]
yonehot=keras.utils.to_categorical(Y)
xfloat=X.astype('float32')

# Split training and testing data
trainx, testx, trainy, testy=ttsplit(xfloat, yonehot, test_size=.2)


