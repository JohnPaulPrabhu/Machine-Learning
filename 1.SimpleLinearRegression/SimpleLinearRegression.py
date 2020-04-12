# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:29:44 2020

@author: Johnpaul.Raj
"""
'''
Salary Prediction

https://medium.com/analytics-vidhya/machine-learning-project-1-predict-salary-using-simple-linear-regression-d83c498d4e05
'''
import pandas as pd
import numpy as np
'''Step 1'''
# Load the dataset into a file
dataset = pd.read_csv('D:\PERSONAL DATA\Studies\Excel\Machine Learning Projects\machine_learning-master\project_1_simple_linear_regression\Salary_Data.csv')

# Seperate the indepandent and dependent varialbe
x=dataset.iloc[: , :-1].values
y=dataset.iloc[: , 1].values

'''Step 2'''
# Spliting the data into test and training data
# We will use the training dataset for training the model and 
# then check the performance of the model on the test dataset.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state= 0)

'''Step 3'''
# Fit simple linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

'''Step 4'''
# Predict the test set
# Using the regressor we trained in the previous step, we will now use it 
# to predict the results of the test set and compare the predicted values with the actual values
y_pred = regressor.predict(x_test)

'''Step 5'''
# Visualinzing the training set
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience")
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()

'''Step 6'''
# Visualinzing the test set
plt.scatter(x_test, y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()

'''Step 7'''
# Make new Prediction
# We can also make brand new predictions for data points that do not exist in the dataset. 
# Like for a person with 15 years experience

new_salary_pred = regressor.predict([[15]])





