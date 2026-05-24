# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the student placement dataset and select independent variables (X) and placement status as dependent variable (Y). 
2.Split the dataset into training data and testing data using train_test_split(). 
3.Create the LogisticRegression() model and train it using fit(X_train, y_train).
4.Predict placement status using test data, calculate accuracy, and display the result.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUJITH MANO M
RegisterNumber:  212225220109
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("Placement_Data.csv")
data = data.drop("salary", axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
X1 = X.iloc[:, 0].values.reshape(-1, 1)
model_plot = LogisticRegression(max_iter=1000)
model_plot.fit(X1, y)
plt.scatter(X1, y, color='blue')
x_values = np.linspace(X1.min(), X1.max(), 100)
y_values = model_plot.predict_proba(x_values.reshape(-1,1))[:,1]

plt.plot(x_values, y_values)

plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression Curve")
plt.show()
```


## Output:
<img width="848" height="620" alt="image" src="https://github.com/user-attachments/assets/c0fdff4f-9318-49cd-b7b3-7988b21804a5" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
