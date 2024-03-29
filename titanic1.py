#Titanic Tutorial
#Data wrangling,  ujy
import pandas as pd 
import numpy as np 
import random as rnd 

#These are the viz module
import seaborn as sns 
import matplotlib.pyplot as plt 

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

#Combine the two dataframes
combine = [train_df, test_df]

#print the columns of the trainig dataset
print(train_df.columns.values)

#print the head
print(train_df.head)
#print the tail
print(train_df.tail)
train_df.info()
test_df.info()
#Here we are trying to assess how representative the traing set is by taking 40% of it?
print('_'*40)
    


#Decsribe the traing set
print(train_df.describe()) 

#Print and describe the datatypes
print(train_df.describe(include=['O']))


print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

plt.show() 

#3rd Class had the most passengers but most died
#Infants in 2nd and 3rd class mostly survived
#1st class passenger survived...shockingly :-)




grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()



