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

<img width="826" height="311" alt="Screenshot 2026-04-28 140901" src="https://github.com/user-attachments/assets/e651c910-0878-4528-8a05-0dba292cecae" />


X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)

<img width="258" height="826" alt="Screenshot 2026-04-28 140943" src="https://github.com/user-attachments/assets/898b7127-a666-419a-900c-cba765c483c7" />


print(X1_Scaled)

<img width="378" height="818" alt="Screenshot 2026-04-28 141008" src="https://github.com/user-attachments/assets/4026a30f-54aa-4b38-bec2-c543651b9c65" />


theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

<img width="286" height="48" alt="Screenshot 2026-04-28 141028" src="https://github.com/user-attachments/assets/6c96628b-9fd5-456e-b15a-ecff9439e757" />






```




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
