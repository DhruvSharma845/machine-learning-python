#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:55:53 2019

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


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)
forest_clf.fit(X_train_std, y_train)

from sklearn.metrics import accuracy_score
y_pred = forest_clf.predict(X_test_std)
print('Accuracy Score: %.2f' % accuracy_score(y_test, y_pred))
