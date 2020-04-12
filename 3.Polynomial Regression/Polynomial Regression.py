# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:02:20 2020

@author: Johnpaul.Raj
"""
import pandas as pd
import numpy as np

'''Step 1'''
'''Load the data into the dataset'''
dataset = pd.read_csv('D:\PERSONAL DATA\Studies\Excel\Machine Learning Projects\machine_learning-master\project_3_polynomial_regression/Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

'''
Step 2:
    Fit simple linear regression model to training set
'''
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x,y)

'''
Step 3: 
    Visualize Linear Regression Results
'''
import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.plot(x,lr.predict(x))
plt.title("Linear Regression")
plt.xlabel('Level')
plt.ylabel("Salary")
plt.show()

'''
Step 4: 
    Predict Linear Regression Results
'''
lr.predict([[6.5]])

'''Now let's check the predictions by polynomial regression
'''
'''
Step 5: 
    Convert X to polynomial format
'''
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=3)
x_poly = pf.fit_transform(x)

'''
Step 6: 
    Fitting Polynomial Regression
'''
lr2=LinearRegression()
lr2.fit(x_poly,y)

'''
Step 7: 
    Visualize Linear Regression Results
'''
plt.scatter(x,y,color = 'red')
plt.plot(x,lr2.predict(pf.fit_transform(x)))
plt.title("Poly Regression - Degree 2") 
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()

'''
Step 8: 
    Predict Polynomial Regression Results
'''
New_Salary_pred = lr2.predict(pf.fit_transform([[6.5]]))
print("New predicted salary is",New_Salary_pred)
'''New predicted salary is [189498.10606061]'''
'''We get a prediction of $189k. It is not too bad.
 But let us increase the degree and see if we get better results.'''

'''
Step 9:
    Change degree to 3 and run steps 5–8
'''
pf=PolynomialFeatures(degree=3)
x_poly = pf.fit_transform(x)

lr2=LinearRegression()
lr2.fit(x_poly,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,lr2.predict(pf.fit_transform(x)))
plt.title("Poly Regression - Degree 3") 
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()

New_Salary_pred = lr2.predict(pf.fit_transform([[6.5]]))
print("New predicted salary is",New_Salary_pred)

'''
Step 10:
    Change degree to 4 and run steps 5–8
'''
pf=PolynomialFeatures(degree=4)
x_poly = pf.fit_transform(x)

lr2=LinearRegression()
lr2.fit(x_poly,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,lr2.predict(pf.fit_transform(x)))
plt.title("Poly Regression - Degree 4") 
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()

New_Salary_pred = lr2.predict(pf.fit_transform([[6.5]]))
print("New predicted salary is",New_Salary_pred)

'''
So in this case by using Linear Regression — 
    we got a prediction of $330k and by using Polynomial Regression we got a 
    prediction of 158k.
'''
'''
Tring to increase the degree to 5
'''
pf=PolynomialFeatures(degree=5)
x_poly = pf.fit_transform(x)

lr2=LinearRegression()
lr2.fit(x_poly,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,lr2.predict(pf.fit_transform(x)))
plt.title("Poly Regression - Degree 5") 
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()

New_Salary_pred = lr2.predict(pf.fit_transform([[6.5]]))
print("New predicted salary is",New_Salary_pred)

'''
New predicted salary is [174878.07765118]
But the predicted value is way higher than what we need to get
So we are keeping the degree as 4
So we will get the best possible value
'''
