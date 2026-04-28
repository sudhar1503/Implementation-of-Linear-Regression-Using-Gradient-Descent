# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program and output:
```
Developed by: SUDHARSAN U
RegisterNumber:  212225040433


import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()

<img width="651" height="243" alt="image" src="https://github.com/user-attachments/assets/62edbf65-3314-4ad0-a84d-4e3c02fda19d" />

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)

<img width="271" height="871" alt="image" src="https://github.com/user-attachments/assets/5d46629c-d0c6-4699-a34a-a012f0eb43e4" />

print(X1_Scaled)

<img width="397" height="869" alt="image" src="https://github.com/user-attachments/assets/337b42e7-738a-403a-bb06-0b38bceb7280" />

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

<img width="294" height="50" alt="image" src="https://github.com/user-attachments/assets/aba95e7d-89e7-4326-8c7c-d0f013b6b725" />





```




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
