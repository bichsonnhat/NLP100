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

columns = ('CATEGORY', 'URL')

train = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/train.txt', names = columns, sep = '\t')
valid = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/valid.txt', names = columns, sep = '\t')
test = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/test.txt', names = columns, sep = '\t')

# confuse ...
lableINT = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = train['CATEGORY'].map(lableINT) 
y_valid = valid['CATEGORY'].map(lableINT)
y_test = test['CATEGORY'].map(lableINT)
# del train, valid, test

x_train = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/train.feature.txt',
                      sep='\t', header=None)
x_valid = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/valid.feature.txt',
                      sep='\t', header=None)
x_test = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/test.feature.txt',
                     sep='\t', header=None)

# error ?
lg = LogisticRegression(class_weight='balanced')
lg.fit(x_train, y_train)
# lr = LogisticRegression(class_weight= 'balance')
# lr.fit(x_train, y_train)
# df = pd.read_csv('./NewsAggregatorDataset/newsCorpora.csv',
#                 header=None,
#                 sep='\t',
#                 names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
# )

# # Synthetic Minority Oversampling Technique (SMOTE):
# # https://learn.theprogrammingfoundation.org/getting_started/intro_data_science/module4#synthetic-minority-oversampling-technique-smote

# # Independent variables
# X = df.drop(columns = ['ID'])
# # Dependent variables
# Y = df[['ID']]

# #Smote sampling
# os = SMOTE(random_state = 0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
# columns = X_train.columns

# smote_data_X, smote_data_Y = os.fit_resample(X_train, Y_train)
# smote_data_X = pd.DataFrame(data = smote_data_X, columns=columns)
# smote_data_Y = pd.DataFrame(data = smote_data_Y, columns = ['ID'])
# # train, train_test = train_test_split(df, test_size = 0.2, shuffle = True, random_state = 42,  stratify = df['CATEGORY'])

# # train, train_test = train_test_split(df, test_size = 0.2, shuffle = True, random_state = 42,  stratify = df['CATEGORY'])
# # valid, valid_test = train_test_split(df, test_size = 0.9, shuffle = True, random_state = 42,  stratify = df['CATEGORY'])
# # test, test_test = train_test_split(df, test_size = 0.9, shuffle = True, random_state = 42,  stratify = df['CATEGORY'])
# print(logTrain(X_train, Y_train));
