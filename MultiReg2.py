# Gradient Descent for Multiple Linear Regression
# Let yhat be the estimated y value (y prediction)
# Let w be beta coefficient
# yhat = np.dot(w,x) + b
# N = Number of training examples
# loss function  = (sum(y-yhat)**2)/2*N
# Let dldw be partial derivative of loss function with respect to parameter w

import numpy as np
import pandas as pd


#Import Data
df = pd.read_csv("C:/Users/raptu/Desktop/PythonWorks/Housing_Price.csv")
print(df)
bedrooms = df['total_bedrooms']
df = df.drop(['total_bedrooms','longitude', 'latitude', 'ocean_proximity', 'gender', 'housing_median_age', 'median_income', 'median_house_value'], axis=1)
df['total_bedrooms'] = bedrooms
df['households'] = pd.to_numeric(df['households'], errors='coerce')
df = df.fillna(0)
df = df.astype(float)
print(df.dtypes)

# Initialise some parameters
df_np = df.to_numpy()
x = df_np[:, :3]
y = df_np[:, -1]


#Parameters
w = np.zeros([3,1])
b = 0

#Hyperparameter
learning_rate = 10

#Create gradient descent function
def gdescent (x,y,w,b,learning_rate):
    dldw = np.zeros_like(w)
    dldb = 0.0
    N = x.shape[0]


    for xi,yi in zip(x,y):
        yhat = np.dot(xi,w) + b
        loss = yhat - yi
        dldw += xi.reshape(-1,1)*loss
        dldb += loss

    #Update the w parameter
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    

    return w,b


for epoch in range(50):
    w,b = gdescent(x,y,w,b,learning_rate)
    yhat = np.dot(x,w) + b
    loss = yhat - y
    j_wb = np.divide(np.sum((loss)**2, axis=0), x.shape[0])
    print(f'{epoch} Error is {loss}, Cost function is {j_wb} Beta coefficients are:{w}, bias is:{b}')
    #Run gradient descent
    pass
