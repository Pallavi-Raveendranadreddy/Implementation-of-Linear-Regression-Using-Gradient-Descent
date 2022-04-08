# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Use the standard libraries in python for Gradient Design.
2.Upload the dataset and check any null value using .isnull() function. 3.Declare the default values for linear regression.
3.Calculate the loss usinng Mean Square Error.
4.Predict the value of y.
5.Plot the graph respect to hours and scores using scatter plot function.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:V.Pallavi 
RegisterNumber:212221240059

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('scores.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()  
*/
```

## Output:
![linear regression using gradient descent](https://github.com/Pallavi-Raveendranadreddy/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/78c2acd792b15999e38056709a1ae7381431011b/lr.PNG)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
