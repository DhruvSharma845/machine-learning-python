#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Linear Neuron (Adaline)
Objective function: Sum of squared errors
Created on Sat Sep 21 14:32:59 2019

@author: dhruv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# m : num of training examples
# n : num of features
class Adaline(object):
    
    def __init__(self, learning_rate = 0.01, epochs = 50, random_state = 1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
    
    def activation(self, X):
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0 , 1, -1) 
        
    # X: (m x n)
    # y: (m x 1)
    def fit(self, X, y):
        
        rgen = np.random.RandomState(self.random_state)
        
        # w : (vector of size n)
        self.w = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1])
        self.b = 0.0
        
        self.cost = []
        for epoch in range(self.epochs):
            # output : vector of size m
            output = self.activation(X)
            errors = y - output
            dw = np.dot(X.T,errors)
            self.w += self.learning_rate * dw
            self.b += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)

# Learning model on Iris Dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None) 
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1 ,1)
X = df.iloc[0:100, [0,2]].values       
print(X)
print(y)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada_clf_1 = Adaline(epochs=10, learning_rate=0.01)
ada_clf_1.fit(X, y)
ax[0].plot(range(1, len(ada_clf_1.cost) + 1), np.log10(ada_clf_1.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada_clf_2 = Adaline(epochs=10, learning_rate=0.0001)
ada_clf_2.fit(X,y)
ax[1].plot(range(1, len(ada_clf_2.cost) + 1), ada_clf_2.cost, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()