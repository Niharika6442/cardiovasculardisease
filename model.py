#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle



data=pd.read_csv('/Users/Niharika/Desktop/Data Science/untitled folder/CardioVascularDisease/cardio_train.csv',delimiter=';',)
data=data.drop(columns=['id'],axis=1)
data['age']=data['age'].apply(lambda x:x/365)
data.rename(columns={'ap_hi':'systolic','ap_lo':'diastolic','cardio':'disease'},inplace=True)

X= data.drop(columns=['disease'],axis=1)
y=data['disease']

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=40, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=19,
                       n_jobs=None, oob_score=False, random_state=123,
                       verbose=0, warm_start=False)
rf.fit(X,y)


pickle.dump(rf, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))

