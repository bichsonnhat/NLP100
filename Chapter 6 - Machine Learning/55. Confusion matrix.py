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
from sklearn.metrics import accuracy_score

def get_option():
    argparser = ArgumentParser()

    argparser.add_argument('--corpus', default='C:/Users/ADMIN/Desktop/NewsAggregatorDataset/newsCorpora.csv')
    
    argparser.add_argument('--train', default='train.txt')
    argparser.add_argument('--valid', default='valid.txt')
    argparser.add_argument('--test', default='test.txt')
    
    argparser.add_argument('--train_feature', default='train.feature.txt')
    argparser.add_argument('--valid_feature', default='valid.feature.txt')
    argparser.add_argument('--test_feature', default='test.feature.txt')
    
    argparser.add_argument('--result', default='result.png')

    return argparser.parse_args()

def train(X_train, Y_train):
    lg = LogisticRegression(random_state=0, max_iter=10000)
    lg.fit(X_train, Y_train)

    return lg

def predict(lg, X):
    return ([np.max(lg.predict_proba(X), axis=1), lg.predict(X)])

def main():
    args = get_option()

    df_train = pd.read_csv(args.train, sep='\t')
    df_valid = pd.read_csv(args.valid, sep='\t')
    df_test = pd.read_csv(args.test, sep='\t')

    X_train = pd.read_csv(args.train_feature, sep='\t')
    X_valid = pd.read_csv(args.valid_feature, sep='\t')
    X_test = pd.read_csv(args.test_feature, sep='\t')

    lg = train(X_train, df_train['CATEGORY'])
 
    train_pred = predict(lg, X_train)
    test_pred = predict(lg, X_test)

    train_cm = confusion_matrix(df_train['CATEGORY'], train_pred[1])
    test_cm = confusion_matrix(df_test['CATEGORY'], test_pred[1])

    print("Confusion Matrix of Train:")
    print(train_cm)
    print("Confusion Matrix of Test:")
    print(test_cm)

if __name__ == '__main__':
    main()
