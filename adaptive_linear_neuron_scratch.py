#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Linear Neuron (Adaline)
Objective function: Sum of squared errors
Created on Sat Sep 21 14:32:59 2019
Last Modified on Sun Nov 03 18:12:00 2019

@author: Dhruv
"""
import numpy as np

"""
 n : num of training examples
 m : num of features
 NOTE: Instead of comparing the prediction with true class labels,
 it compares the true class labels with the activation values
"""


class Adaline(object):

    def __init__(self, learning_rate=0.01, epochs=50, random_state=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.w = None
        self.b = None
        self.cost = []

    """
        finds the activation values for the input instances
    """
    def activation(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    """
        method to train the model on training instances
        X : input features ; shape = (n x m)
        y : class labels ; shape = (n x 1)
    """
    def fit(self, X, y):
        randomGenerator = np.random.RandomState(self.random_state)

        # w : (vector of size m)
        self.w = randomGenerator.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = 0.0

        for epoch in range(self.epochs):
            # output : vector of size m
            output = self.activation(X)

            errors = y - output
            dw = np.dot(X.T, errors)
            self.w += self.learning_rate * dw
            self.b += self.learning_rate * errors.sum()

            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)

        return self.cost
