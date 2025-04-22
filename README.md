# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.
7. Print the results.

## Program:
```python
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by  : SANJAY M
RegisterNumber: 212223230187 
```
```PYTHON
import pandas as pd
import numpy as np
df=pd.read_csv("Placement_Data.csv")
df
df=df.drop("sl_no",axis=1)
df=df.drop("salary",axis=1)
df.head()
df["gender"]=df["gender"].astype("category")
df["ssc_b"]=df["ssc_b"].astype("category")
df["hsc_b"]=df["hsc_b"].astype("category")
df["degree_t"]=df["degree_t"].astype("category")
df["workex"]=df["workex"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["status"]=df["status"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df.dtypes
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
## display dependent variables
y
theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,Y):
    h=sigmoid(x.dot(theta))
    return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-Y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)
print(y_pred)
print(y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## Output:
### Placement Dataset
![image](https://github.com/user-attachments/assets/c5615067-496d-4bda-aac2-b27b17dc58a6)

### Dataset after Feature Engineering
![image](https://github.com/user-attachments/assets/2513b14a-15e7-4c97-9956-748dbf155eb3)

### Datatypes of Feature column
![image](https://github.com/user-attachments/assets/4efbd32a-c77e-4990-b141-e5ad63c748f6)

### Dataset after Encoding
![image](https://github.com/user-attachments/assets/a3b5fef9-7db6-44c3-bfe9-b1e09ba50f1d)

### Y Values
![image](https://github.com/user-attachments/assets/9dc80f65-7ff0-4735-bbf5-135258c99d31)

### Accuracy
![image](https://github.com/user-attachments/assets/08efa55b-c241-44e5-af12-2693bf3229ff)

### Y Predicted
![image](https://github.com/user-attachments/assets/b4d1297a-3cd4-4b9c-88c5-74ba337c1f19)

### Y Values
![image](https://github.com/user-attachments/assets/b7f798b6-ca07-489f-ba88-67a26bc06a58)

### Y Predicted with different X Values
![image](https://github.com/user-attachments/assets/973faeeb-a164-483d-9fe7-cfc560aed7b9)
![image](https://github.com/user-attachments/assets/c0f43dc2-cd34-46a2-b83e-46e9072c1f8b)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

