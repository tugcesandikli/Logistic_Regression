#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# In[51]:


data = pd.read_csv("/Users/tugcesandikli/Downloads/data.csv")


# In[52]:


data.head()


# In[53]:


data.info()


# In[54]:


data.drop(["Unnamed: 32","id"],axis=1, inplace=True)


# In[55]:


data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]


# In[56]:


y = data.diagnosis.values   # Values metodu ile numpy array'e çevrilir
x_data = data.drop(["diagnosis"],axis=1)


# In[57]:


x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[58]:


x


# In[59]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.20,random_state=42)


# In[60]:


x_train = x_train.T
x_train.shape


# In[61]:


x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# In[62]:


print(x_test.shape)


# In[63]:


y_train.shape


# In[64]:


y_test.shape


# In[65]:


def initialize_weights_and_bias(dimesion):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


# In[66]:


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# In[67]:


sigmoid(0)


# In[68]:


def forward_backward_propagation(w,b,x_train,y_train):
    
    # Forward Propagation
    z = np.dot(w,T,x_train) + b
    y_head= sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    # Backward Propagation
    derivative_weight = (np.dot(x_train,((y_head-y,train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,gradients


# In[69]:


def update(w, b, x_train, learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iteration):
        cost,gradients = forward_backwarkd_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
            
    parameters = {"weight": w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# In[70]:


def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w,T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))   #(1,114)
    
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
        
    return Y_prediction


# In[71]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 3, num_iterations = 300)


# In[72]:


from sklearn import linear_model
lr = linear_model.LogisticRegression(random_state=42,max_iter=40)
lr.fit(x_train.T,y_train.T)


# In[73]:


y_pred = lr.predict(x_test.T)


# In[25]:


print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))

