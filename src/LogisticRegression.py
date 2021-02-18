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





def main():
    print('main')
    df_train = pd.read_csv(osp.join('..' , 'data' , '20_News' , '20_News_Train_Preprocessed.csv') , index_col=False)
    df_test = pd.read_csv(osp.join('..', 'data', '20_News', '20_News_Test_Preprocessed.csv'), index_col=False)



    cv = CountVectorizer(max_df=0.5, min_df=2, max_features=300)

    print(df_train['Text'])

    X_train = cv.fit_transform(df_train['Text']).toarray()
    Y_train = df_train['Class']
    #X_train = SelectKBest(chi2, k=5000).fit_transform(X_train, Y_train)
    print(cv.get_feature_names())
    print(X_train[1][:-1])








if __name__ == '__main__':
    main()

