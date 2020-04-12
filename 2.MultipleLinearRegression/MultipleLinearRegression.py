# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:21:32 2020

@author: Johnpaul.Raj
"""
'''
Machine Learning Project 2: 
    Predict Profit using Multiple Linear Regression
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Step 1'''
'''Load the data into the dataset'''
dataset=pd.read_csv('D:\PERSONAL DATA\Studies\Excel\Machine Learning Projects\machine_learning-master\project_2_multiple_linear_regression/50_Startups.csv')
x=dataset.iloc[: , :-1].values
y=dataset.iloc[: , 4].values

print(pd.get_option('display.max_columns'))
'''
Step 2
    Convert the text variable to numbers
    We can see that in our dataset we have a categorical variable which is “State” 
    which we have to encode. Here the “State” variable is at index 3.
    We use LabelEncoder class to convert text to numbers.
'''
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
LabelEncoder_x = LabelEncoder()
x[:,3] = LabelEncoder_x.fit_transform(x[:,3])
'''Now the states have been converted into numbers'''

'''
Step 3:
    Use OneHotEncoder to introduce Dummy variables
    If we leave the dataset in the above state, 
    it will not be right since New York has been assigned a value 2 and 
    California has been assigned 0. So the model might assume New York is higher 
    than California which is not right. which is not right
    So to avoid this we have to introduce dummy variables using OneHotEncoder as shown below.
'''
from sklearn.compose import ColumnTransformer
# oneHotEncoder = OneHotEncoder(categorical_features=[3])
# x=oneHotEncoder.fit_transform(x).toarray()
# We cannot use the above two line since sklearn updated its classess
ColumnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x=np.array(ColumnTransformer.fit_transform(x), dtype=np.int64)

'''
Step 4:
    Remove the dummy trap
'''
x=x[:, 1:]

'''
Step 5:
    Split the dataset into training set and test set 
'''
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state= 0)

'''
Step 6:
    Fit simple linear regression model to training set
'''
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

'''
Step 7:
    Predict the test set
'''
y_pred = lr.predict(x_test)

'''
Step 8:
    Backward Elimination
    
    
    iii) 
'''
# i) First we nedd to add the column of ones to the dataset as the first column

ones = np.ones(shape=(50,1), dtype=int)
x=np.append(arr=ones,values=x, axis=1)

# ii) we will be creating a new optimal matrix of features — we will call it X_opt
#        This will contain only the independent features that are significant in predicting profit.

X_opt = x[:, [0,1,2,3,4,5]]

# iii)Next we need to select a significance level (SL) — 
# here we decide on significance level of 0.05.

'''
iv) we create a new regressor of the OLS class (Ordinary Least Square) from statsmodels library.
It takes 2 arguments
- endog : which is the dependent variable.
- exog : which is the matrix containing all independent variables.
'''
import statsmodels.api as sm
'''First time doing the backward elimination'''
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


'''Scond time doing the backward elimination'''
X_opt= x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

'''Third time doing the backward elimination'''
X_opt= x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

'''Fourth time doing the backward elimination'''
X_opt= x[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

'''Fifth time doing the backward elimination'''
X_opt= x[:,[0,3]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

'''
Finally we are left with only 1 independent variable which is the R&D spent
'''

'''
We can build our model again but this time taking only 1 independent variable
 which is the R&D spent and do the prediction and our 
 results will be better than the first time.
'''











