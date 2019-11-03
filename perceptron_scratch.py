#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perceptron Algorithm
Created on Sat Sep 21 12:56:49 2019
Last Modified on Sun Nov 03 16:38:00 2019

@author: Dhruv
"""

import numpy as np

"""
n : number of training instances
m : number of features
"""


class Perceptron(object):

    def __init__(self, learning_rate=0.01, epochs=50, random_state=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.w = None
        self.b = None
        self.errors = []

    """
        method to train the model using the input data 
        X: input features ; shape = (n x m)
        y: input labels ; shape = (n x 1)
    """

    def fit(self, X, y):

        # initialize the weights
        randomGenerator = np.random.RandomState(self.random_state)
        self.w = randomGenerator.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = 0.0

        for epoch in range(self.epochs):
            """
                for every epoch, iterate on each instance
                and update the weights accordingly
            """
            errorInSingleEpoch = 0
            for x, label in zip(X, y):
                predictedOutput = self.predict(x)
                error = label - predictedOutput

                # update the weights by the following formula:
                # w = w + learning_rate * (label - prediction) * x
                dw = self.learning_rate * error
                self.w += dw * x
                self.b += dw

                errorInSingleEpoch += int(error != 0)

            self.errors.append(errorInSingleEpoch)

        return self.errors

    """
        method to calculate the predicted target 
        from the input instance
    """

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        return np.where(z >= 0.0, 1, -1)
