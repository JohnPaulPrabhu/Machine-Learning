# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:02:40 2020

@author: Johnpaul.Raj
"""
"""
Machine Learning Project 4: Predict Salary using Support Vector Regression
"""

import pandas as pd
import numpy as np

'''Step 1'''
'''Load the data into the dataset'''
dataset=pd.read_csv('D:\PERSONAL DATA\Studies\Excel\Machine Learning Projects\machine_learning-master\project_4_support_vector_regression/Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:].values

'''
Step 2:
    "Feature Scaling"
    SVR class does not apply feature scaling in its algorithm. 
    So we have to apply feature scaling.
'''
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

'''
Step 3:
    "Fit SVR"
    We will be using the SVR class from the library sklearn.svm.
    First we create an object of the SVR class and pass
    kernel parameter as “rbf” (Radial Basis Function) and 
    then call the fit method passing the X and y.
'''
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

'''
Step 4:
    Visualization
'''
import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('SVR')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

'''
Step 5:
    "Make Predictions"
We want to predict the salary for an employee at level 6.5 . 
Now since we applied feature scaling to X and y — 
first we will have to do feature scaling to transform value 6.5
Then we have to do the prediction.
Finally since the predicted value is already scaled, 
we have to do inverse transformation to get the actual value.
'''
# First transform 6.5 to feature scaling
sc_x_val = sc_x.transform(np.array([[6.5]]))
# Second predict the value
scaled_y_pred = regressor.predict(sc_x_val)
# Third - since this is scaled - we have to inverse transform
y_pred = sc_y.inverse_transform(scaled_y_pred)

print("The predicted value is",y_pred)














