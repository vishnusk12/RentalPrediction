# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:24:43 2017

@author: Vishnu
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from math import sqrt
from sklearn.metrics import mean_squared_error

activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']
#hidden_layer_sizes = range(100,1000)
#max_iter =  range(100,1000)

excel = pd.ExcelFile("C:/Users/hp/Documents/Python Scripts/Rental_Prediction/Sample_Rental_Data_20171121.xlsx")
sheets = excel.sheet_names
for i in sheets:
    df = excel.parse(i)
#    df = df.fillna(lambda x: x.median())
    df = df.dropna()
    df = df.apply(LabelEncoder().fit_transform)
    X = df.drop('Face Rental', axis=1).values
    y = df['Face Rental'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    params_Grid_MLP = dict(activation=activation, 
                           solver=solver, 
                           learning_rate=learning_rate)
    Grid_MLP = GridSearchCV(MLPRegressor(), params_Grid_MLP)
#    reg = MLPRegressor()
#    model = RFE(reg)
#    reg.fit(X_train, y_train)
    Grid_MLP.fit(X_train, y_train)
    print (Grid_MLP.best_estimator_)
    print (Grid_MLP.best_score_)
    predictions = Grid_MLP.predict(X_test)
#    predictions = reg.predict(X_test)
#    print (model.ranking_)
    error = sqrt(mean_squared_error(y_test, predictions))
    acc = 100 - error
    print (acc)