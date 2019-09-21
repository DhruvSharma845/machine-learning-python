#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:05:52 2019

@author: dhruv
"""

# Binary Classification

# LR from scratch using numpy
import numpy as np
import pandas as pd

# class for logistic regression machine learning algorithm
# n : number of features
# m : number of training examples
class LogisticRegression(object):
    
    # constructor to pass initial values for the model building 
    def __init__(self, learning_rate = 0.001, epochs = 50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    # fit the model using training data
    # finding w and b that minimize the cost function
    # X_train : input data with training examples stacked up column wise(n x m)
    # y_train : input labels stacked up column wise(1 x m)
    def fit(self, X, y):
        
        # number of training examples 
        m = X.shape[1]
        
        # parameters of the algorithm
        # w : n x 1 numpy array
        # b : 1 x 1 numpy array
        self.w = np.zeros((X.shape[0], 1))
        self.b = np.zeros((1,1))
        
        self.costs = []
        
        for epoch in range(self.epochs):
        
            # calculating activation
            # output : vector of size m
            output = self.activation(X)
        
            # calculating loss function value over the predictions
            cost = self.cost_fn(y, output)
        
            self.costs.append(cost)
        
            # calculating gradients
            # dz = output - y_train
            dz = output - y
            dw = 1/m * np.matmul(X, dz.T) 
            db = 1/m * np.sum(dz)
        
            #updating parameters
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        
    # return the value of sigmoid function 
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    # defining the loss function
    def cost_fn(self, y, output):
        L = - np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        return L
    
    # calculates the activation function value
    def activation(self, X):
        return self.sigmoid(np.matmul(self.w.T, X) + self.b)
    
    # predict the target value using the built model
    def predict(self, X):
        return np.where(self.activation(X) >= 0.5, 1, 0)    
    
    
# Learning model on Iris Dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None) 
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0 ,1)
X = df.iloc[0:100, [0,2]].values       
print(X)
print(y)
print('Class labels: ', np.unique(y))

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr_clf = LogisticRegression(learning_rate=0.01, epochs=15)
lr_clf.fit(X_train_std.T, y_train)

y_pred = lr_clf.predict(X_test_std.T)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

