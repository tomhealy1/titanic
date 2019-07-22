#Titanic Tutorial
#Data wrangling,  ujy
import pandas as pd 
import numpy as np 
import random as rnd 

#These are the viz module
import seaborn as sns 
import matplotlib.pyplot as pyplot

#I will run this from this file and then add separete files for each model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#We have two data sets here, one for training and one testing. I add and "r" to the path to make it accessible
train_df = pd.read_csv(r"C:\Users\Teamwork\Desktop\titanic\train.csv")
test_df = pd.read_csv(r"C:\Users\Teamwork\Desktop\titanic\test.csv")

combine = [train_df, test_df]

print(train_df.columns.values)

print(train_df.head)

print(train_df.tail)
train_df.info()
test_df.info()

print('_'*40)

print(train_df.describe()) 


print(train_df.describe(include=['O']))