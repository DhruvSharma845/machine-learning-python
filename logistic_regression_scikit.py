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
import matplotlib.pyplot as plt

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sc = StandardScaler()

lr_clf = LogisticRegression(C=1.0, random_state=1, solver='lbfgs', multi_class='auto')

pipe_lr_clf = Pipeline([
        ('std_scaler', sc),
        ('log_regression', lr_clf)
        ])


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=1)

scores = []
for k, (train,holdout) in enumerate(kfold.split(X_train, y_train)):
    pipe_lr_clf.fit(X_train[train], y_train[train])
    score = pipe_lr_clf.score(X_train[holdout], y_train[holdout])
    scores.append(score)
    print('Fold: %d, Class dist: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

print('CV accuracy: %.3f' % (np.mean(scores)))

pipe_lr_clf.fit(X_train, y_train)
y_pred = pipe_lr_clf.predict(X_test)
print('Accuracy Score: %.2f' % pipe_lr_clf.score(X_test, y_test))

# learning curves
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipe_lr_clf,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1,1.0,10.0),
        cv=10,
        n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,color='blue', marker='o',markersize=5, label='training accuracy')
plt.fill_between(train_sizes,train_mean + train_std,train_mean - train_std,alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()


# Grid Search
from sklearn.model_selection import GridSearchCV
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = {'log_regression__C': param_range}
gs = GridSearchCV(estimator=pipe_lr_clf, param_grid = param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
