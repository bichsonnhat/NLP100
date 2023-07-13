from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# from imblearn.over_sampling import SMOTE
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# import statsmodels.api as sm
from argparse import ArgumentParser

def predict(lg, X):
    return ([np.max(lg.predict_proba(X), axis=1), lg.predict(X)])
columns = ('CATEGORY')

train = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/train.txt', names = columns, sep = '\t')
valid = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/valid.txt', names = columns, sep = '\t')
test = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/test.txt', names = columns, sep = '\t')

# confuse ...
lableINT = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = train['CATEGORY'].map(lableINT) 
y_valid = valid['CATEGORY'].map(lableINT)
y_test = test['CATEGORY'].map(lableINT)
del train, valid, test

x_train = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/train.feature.txt',
                      sep='\t', header=None)
x_valid = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/valid.feature.txt',
                      sep='\t', header=None)
x_test = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/test.feature.txt',
                     sep='\t', header=None)

# error ?
lg = LogisticRegression(class_weight='balanced')
lg.fit(x_train, y_train)

train_predict = predict(lg, x_train)
test_predict = predict(lg, x_test)
