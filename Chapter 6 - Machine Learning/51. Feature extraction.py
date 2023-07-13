import pandas as pd
import string
import re
# from ex50 import get_option
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# read info
df = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/newsCorpora.csv',
                header=None,
                sep='\t',
                names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
)
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['URL', 'CATEGORY', 'HOSTNAME']]
# print(df)
# load current(train, valid, test) csv 
# try to get diffirent ways 
columns = ['URL', 'CATEGORY', 'HOSTNAME']
x_train = pd.read_csv('./train.txt', names = columns, sep = '\t')
x_valid = pd.read_csv('./valid.txt', names = columns, sep = '\t')
x_test = pd.read_csv('./test.txt', names = columns, sep = '\t')

x_train.to_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/train.feature.txt',
               sep='\t', header=False, index=False)
x_valid.to_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/valid.feature.txt',
               sep='\t', header=False, index=False)
x_test.to_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/test.feature.txt',
              sep='\t', header=False, index=False)
# ---------------------------------------------------------------------------------------------------------------------------
# def preprocessing(text):
#     table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
#     text = text.translate(table)
#     text = text.lower()
#     text = re.sub(r'\d', '0', text)
#     return text

# concat combines two Series, also, ... 'dtype = object'
# 'Combine DataFrame objects horizontally along the x axis by passing in axis = 1' else axis = 0
# ignore_index = True(0, 1, 2, ..., n) | False (0, 1, 2, ..., 0, 1, 2...)
# df = pd.concat([train, valid, test], axis = 0, ignore_index = True)
# df['URL'] = df['URL'].map(lambda x: preprocessing(x))
 
# vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

# X_train_valid_test = vectorizer.fit_transform(df['URL'])
# X_train_valid_test = pd.DataFrame(X_train_valid_test.toarray(), columns=vectorizer.get_feature_names()) 

# X_train_valid = X_train_valid_test[:len(df_train) + len(df_valid)]
# X_train = X_train_valid[:len(df_train)]
# X_valid = X_train_valid[len(df_train):]
# X_test = X_train_valid_test[len(df_train) + len(df_valid):]

# X_train.to_csv('./train.txt', sep='\t', index=False)
# X_valid.to_csv('./valid.txt', sep='\t', index=False)
# X_test.to_csv('./test.txt', sep='\t', index=False)
# print((columns))