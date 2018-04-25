# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:10:42 2017

@author: Vishnu
"""

import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

excel = pd.ExcelFile("C:/Users/hp/Documents/Python Scripts/Rental_Prediction/Sample_Rental_Data_20171121.xlsx")
sheets = excel.sheet_names
for i in sheets:
    df = excel.parse(i)
    df = df.dropna()
    df = df.apply(LabelEncoder().fit_transform)
    X = df.drop('Face Rental', axis=1).values
    y = df['Face Rental'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    regressor = [
            SVR(kernel='rbf', gamma=0.7, C=1),
            linear_model.Ridge (alpha = .5),
            linear_model.Lasso(alpha = 0.1),
            linear_model.LassoLars(alpha=.1),
            linear_model.BayesianRidge(),
            MLPRegressor(),
            DecisionTreeRegressor(),
            KernelRidge(),
            PassiveAggressiveRegressor(),
            RANSACRegressor(),
            TheilSenRegressor(),
            RandomForestRegressor()
            ]
    
    result_cols = ["Regressor", "Accuracy"]
    result_frame = pd.DataFrame(columns=result_cols)
    
    for model in regressor:
        name = model.__class__.__name__
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        error = sqrt(mean_squared_error(y_test, predictions))
        acc = 100 - error
        print (name+' accuracy = '+str(acc)+'%')
        acc_field = pd.DataFrame([[name, acc]], columns=result_cols)
        result_frame = result_frame.append(acc_field)
    
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Regressor', data=result_frame, color="r")
    
    plt.xlabel('Accuracy %')
    plt.title('Regressor Accuracy')
    plt.show()  