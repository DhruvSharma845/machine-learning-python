#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perceptron Algorithm
Created on Sat Sep 21 12:56:49 2019

@author: dhruv
"""
import numpy as np
import pandas as pd

class Perceptron(object):
    
    def __init__(self, learning_rate = 0.01, epochs = 50, random_state = 1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
    
    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        return np.where(z >= 0.0, 1, -1)
    
    def fit(self, X, y):
        
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1])
        self.b = 0.0
        
        for epoch in range(self.epochs):
            
            for x, target in zip(X, y):
                output = self.predict(x)
                error = target - output
                dw = self.learning_rate * error
                self.w += dw * x
                self.b += dw
                

# Learning model on Iris Dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None) 
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1 ,1)
X = df.iloc[0:100, [0,2]].values       
print(X)
print(y)


ppn_clf = Perceptron(learning_rate = 0.1, epochs = 10)
ppn_clf.fit(X, y)

#Testing
print(ppn_clf.predict([5.7, 4.1]))
