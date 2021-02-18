import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.feature_selection import SelectKBest  , chi2
import os.path as osp
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from nltk.corpus import wordnet as wn




df_train = pd.read_csv(osp.join('..', 'data', '20_News', '20_News_Train.csv'), index_col=False)
df_test = pd.read_csv(osp.join('..', 'data', '20_News', '20_News_Test.csv'), index_col=False)

# PREPROCESSING
stop = stopwords.words('english')
# tokenize
df_train['Text'] = df_train['Text'].astype(str)
df_train['Text'] = [entry.lower() for entry in df_train['Text']]
df_train['Text'] = [word_tokenize(entry) for entry in df_train['Text']]

df_test['Text'] = df_test['Text'].astype(str)
# print(df_test['Text'])

df_test['Text'] = [entry.lower() for entry in df_test['Text']]
df_test['Text'] = [word_tokenize(entry) for entry in df_test['Text']]

# remove stop words
df_train['Text'] = df_train['Text'].apply(lambda x: [item for item in x if item not in stop])
df_test['Text'] = df_test['Text'].apply(lambda x: [item for item in x if item not in stop])

# Lemmatize according to POS tag

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index, entry in enumerate(df_train['Text']):
    Final_words = []
    lemmatized_word = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word.isalpha():
            word_final = lemmatized_word.lemmatize(word, tag_map[0])
            Final_words.append(word_final)
    df_train.loc[index, 'Text'] = str(Final_words)

for index, entry in enumerate(df_test['Text']):
    Final_words = []
    lemmatized_word = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word.isalpha():
            word_final = lemmatized_word.lemmatize(word, tag_map[0])
            Final_words.append(word_final)
    df_test.loc[index, 'Text'] = str(Final_words)

df_train.to_csv(osp.join('..' , 'data' , '20_News' , '20_News_Train_Preprocessed.csv'))
df_test.to_csv(osp.join('..' , 'data' , '20_News' , '20_News_Test_Preprocessed.csv'))
print('end of process')

