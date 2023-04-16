import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./NewsAggregatorDataset/newsCorpora.csv',
                header=None,
                sep='\t',
                names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
)

df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['URL', 'CATEGORY']]

print(df)

# Randomly shuffle the extracted articles.
# The frac parameter specifies the fraction of rows to return in the random sample. Range [0; 1]
# The random_state parameter is used to ensure reproducibility.
shuffled_df = df.sample(frac = 0.5, random_state = 42)
print(shuffled_df)

# Using 'train_test_split()' to split data
# test_size = 0.8 = 80%
# stratify: An array or Series containing the target variable 
train, train_test = train_test_split(df, test_size = 0.2, shuffle = True, random_state = 42,  stratify = df['CATEGORY'])
# the output of train_test_split() is assigned to a tuple containing two variables: train and valid_test. 
# The train variable is assigned to the training set DataFrame, and the valid_test variable is assigned to the validation/test set DataFrame.
# train is a DataFrame, and valid_test is a DataFrame.

# Note Code: train = train_test_split(df, test_size = 0.8, shuffle = True, random_state = 42,  stratify=df['CATEGORY'])
# train is a tuple containing two DataFrames: the training set and the validation/test set.
# type(train) is list 
valid, valid_test = train_test_split(df, test_size = 0.9, shuffle = True, random_state = 42,  stratify = df['CATEGORY'])
test, test_test = train_test_split(df, test_size = 0.9, shuffle = True, random_state = 42,  stratify = df['CATEGORY'])

# save data
train.to_csv('./train.txt', sep='\t', index=False)
valid.to_csv('./valid.txt', sep='\t', index=False)
test.to_csv('./test.txt', sep='\t', index=False)

# count
# shape return the number of rows and columns in the DataFrame or array, respectively.
print('train', train.shape)
print(train['CATEGORY'].value_counts())
print('\n')
print('valid', valid.shape)
print(valid['CATEGORY'].value_counts())
print('\n')
print('test', test.shape)
print(test['CATEGORY'].value_counts())