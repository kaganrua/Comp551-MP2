from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os.path as osp


twenty_train = fetch_20newsgroups(subset='train' , remove=['headers' , 'footers' , 'quotes'])

print(type(twenty_train.data))
df = pd.DataFrame()

df['Text'] = twenty_train.data
df['Class'] = twenty_train.target

df.to_csv(osp.join( '..','data' , '20_News_Train.csv') , index=False)

twenty_train_test = fetch_20newsgroups(subset='test' , remove=['headers' , 'footers' , 'quotes'])

df_test = pd.DataFrame()

df_test['Text'] = twenty_train_test.data
df_test['Class'] = twenty_train_test.target

df_test.to_csv(osp.join( '..','data' , '20_News_Test.csv') , index=False)






