# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:01:25 2017

@author: Vishnu
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os
import glob

os.chdir('C:/Users/hp/Documents/Python Scripts/Rental_Prediction/')

paths = glob.glob("*.csv")

#for path in paths:
data = pd.read_csv('C:/Users/hp/Documents/Python Scripts/Rental_Prediction/Retail.csv')
del data['Unnamed: 0']
data = data.apply(LabelEncoder().fit_transform)
X = data.drop('Label', axis=1).values
Y = data['Label'].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 1)
clf = RandomForestClassifier()
param_grid = {"n_estimators": [10, 100],
              "criterion": ["gini", "entropy"],
              "max_features": ['auto', 'sqrt', 'log2', 'None'],
              "min_samples_split": [5,10],
              "oob_score": [True, False],
              "bootstrap": [True, False]}
grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=2)
grid_search.fit(x_train, y_train)
predictions = grid_search.predict(x_test)
print('Training Score: %.2f%%' % (grid_search.score(x_train, y_train) * 100))
print('Test Score: %.2f%%' % (grid_search.score(y_train, y_test) * 100))
print (grid_search.best_params_)