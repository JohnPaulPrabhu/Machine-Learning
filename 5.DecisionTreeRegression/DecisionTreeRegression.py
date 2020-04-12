# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:41:54 2020

@author: Johnpaul.Raj
"""
"""
Machine Learning Project 5:
    Predict Salary using Decision Tree Regression
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Step 1:
    Load the Dataset
'''
dataset=pd.read_csv("D:\PERSONAL DATA\Studies\Excel\Machine Learning Projects\machine_learning-master\project_5_decision_tree_regression/Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:].values
'''
Step 2:
    Fit Decision Tree Regressor
'''
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(criterion="mse")
regressor.fit(x,y)

'''
Step 3:
    Visualize
'''
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("DecisionTreeRegressor")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

'''
Step 4:
    Make Prediction
'''
y_pred=regressor.predict([[6.5]])
print("Predicted value",y_pred)


























