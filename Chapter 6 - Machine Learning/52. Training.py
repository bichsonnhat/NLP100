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


# del train, valid, test
def main():
    args = get_option()
    df_train = pd.read_csv(args.train, sep='\t')
    df_valid = pd.read_csv(args.valid, sep='\t')
    df_test = pd.read_csv(args.test, sep='\t')

    X_train = pd.read_csv(args.train_feature, sep='\t')
    X_valid = pd.read_csv(args.valid_feature, sep='\t')
    X_test = pd.read_csv(args.test_feature, sep='\t')

    print(train(X_train, df_train['CATEGORY']))

if __name__ == '__main__':
    main()
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
