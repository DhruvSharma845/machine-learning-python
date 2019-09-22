#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 00:17:34 2019

@author: dhruv
"""
import pandas as pd
import os
import numpy as np

# change the `basepath` to the directory of the
# unzipped movie dataset
basepath = 'aclImdb_v1/aclImdb'

"""
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
df.columns = ['review', 'sentiment']

import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
"""

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')


X_train = df.iloc[:30000, 0].values
X_train = X_train.reshape(30000)
y_train = df.iloc[:30000, 1].values


X_test = df.iloc[30000:, 0].values
X_test = X_test.reshape(20000)
y_test = df.iloc[30000:, 1].values

print('X_train: ',X_train.shape)
print('y_train: ',y_train.shape)
print('X_test: ',X_test.shape)
print('y_test: ',y_test.shape)


from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
#bag_of_words = count.fit_transform(X)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(use_idf=True, lowercase=False, preprocessor=None)
np.set_printoptions(precision=2)
#print(tfidf.fit_transform(bag_of_words))

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

lr_tfidf = Pipeline([
        ('vect', tfidf),
        ('clf', LogisticRegression(random_state=1))
        ])
param_grid = {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [1.0,10.0]
        } 

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)