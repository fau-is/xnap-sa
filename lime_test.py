# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 22:27:04 2020

@author: ov59opom
"""

from sklearn.datasets import load_boston
import sklearn.ensemble
import numpy as np
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

boston = load_boston()

#print (boston['DESCR'])

rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
train, test, labels_train, labels_test = train_test_split(boston.data, boston.target, train_size=0.80)
rf.fit(train, labels_train)

print('Random Forest MSError', np.mean((rf.predict(test) - labels_test) ** 2))

print('MSError when predicting the mean', np.mean((labels_train.mean() - labels_test) ** 2))

categorical_features = np.argwhere(
    np.array([len(set(boston.data[:,x]))
    for x in range(boston.data.shape[1])]) <= 10).flatten()

explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                   feature_names=boston.feature_names, 
                                                   class_names=['price'], 
                                                   categorical_features=categorical_features, 
                                                   verbose=True, mode='regression')

i = 100

exp = explainer.explain_instance(test[i], rf.predict, num_features=5)
exp.show_in_notebook(show_table=True)
