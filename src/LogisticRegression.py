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
    df_train = df_train.drop(df_train.columns[[0]] , axis=1)
    print(df_train)


    tv = TfidfVectorizer(max_df=1., min_df=3, max_features=20000)



    X_train = tv.fit_transform(df_train['Text']).toarray()
    Y_train = df_train['Class']
    #X_train = SelectKBest(chi2, k=3000).fit_transform(X_train, Y_train)
    #print(tv.get_feature_names())
    X_test = df_test['Text']
    X_test = tv.transform(X_test).toarray()
    Y_test = df_test['Class']
    print(X_train)
    clf = LogisticRegression(max_iter=3000)

    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(Y_test , y_pred)
    print(acc)








if __name__ == '__main__':
    main()


