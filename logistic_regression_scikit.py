#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression using scikit learn library
Cost function is based on maximum linkelihood function
Created on Sat Sep 21 16:17:34 2019

@author: dhruv
"""
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print(X.shape)
print(y.shape)
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

from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='auto')
lr_clf.fit(X_train_std, y_train)


probs = lr_clf.predict_proba(X_test_std)
print(probs[:3])

from sklearn.metrics import accuracy_score
y_pred = lr_clf.predict(X_test_std)
print('Accuracy Score: %.2f' % accuracy_score(y_test, y_pred))

