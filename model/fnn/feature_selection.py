#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:13:45 2021

@author: wenrchen
"""

##feature selection with random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from matplotlib import pyplot
import numpy as np
import pandas as pd
import sys
import hydro_index as hi
import retention_model as rm

alphabet='ACDEFGHIKLMNPQRSTVWY'

# define dataset
#X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
df=pd.read_csv(sys.argv[1],sep='\t')
y_object='apex_rt'
max_mo=10800

#feature_rank=[20, 21, 9, 29, 25]
feature_rank=[i for i in range(62)]
feature_size=62

x_attr=sys.argv[3]
y_attr=sys.argv[4]


def compute_X_y(df,x_attr,y_attr,train_index=None):
    df=df.copy()
    y=(df[y_attr].values)/max_mo

    aaAlphabet=sorted(list(alphabet))
    seqs=np.asarray(df[x_attr].values)
    RT=np.asarray(df[y_attr].values)
    
    if(train_index==None):
        custom_index=rm.buildRetentionIndex(aaAlphabet,RT,seqs,True)
        index=dict(zip(alphabet,custom_index))
    else:
        index=train_index
    
    X=[]
    for seq in seqs:
        feature_vector=rm.computeRetentionFeatureVector(alphabet,seq,index)
        X.append(feature_vector)
    X=np.asarray(X)
    
    feature_index=feature_rank[:feature_size]
    X=X[:,feature_index]
    
    print(X.shape)
    
    return X,y,index
    
X,y,train_index=compute_X_y(df,x_attr,y_attr)

df_test=pd.read_csv(sys.argv[2],sep='\t')


X_test,y_test,_=compute_X_y(df_test,x_attr,y_attr,train_index=train_index)

# define the model
model = RandomForestRegressor(n_estimators=50, random_state=0)
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance

#model = LinearSVR(C=0.01, dual=False, loss='squared_epsilon_insensitive',fit_intercept=False,max_iter=5000).fit(X, y)
#importance=model.coef_
print(model.score(X,y))
print(model.score(X_test,y_test))

rank_dict={}
for i,v in enumerate(abs(importance)):
    rank_dict[i]=v
    print('Feature: %0d, Score: %.5f' % (i,v))
rank_sorted=dict(sorted(rank_dict.items(), key=lambda item: item[1],reverse=True))
print(rank_sorted.keys())
## plot feature importance
#pyplot.bar([x for x in range(len(importance))], list(rank_sorted.values()))
#pyplot.show()
