import pandas as pd

df = pd.read_csv('C:/Users/ADMIN/Desktop/NewsAggregatorDataset/newsCorpora.csv',
                header=None,
                sep='\t',
                names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
)

df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['URL', 'HOSTNAME']]

print(df)