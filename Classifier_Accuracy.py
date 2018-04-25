# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:27:53 2017

@author: Vishnu
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('C:/Users/hp/Documents/Python Scripts/Rental_Prediction/Residential.csv')
del data['Unnamed: 0']
data = data.apply(LabelEncoder().fit_transform)
X = data.drop('Label', axis=1).values
Y = data['Label'].values
num = len(list(set(Y)))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state = 42)


result_cols = ["Classifier", "Accuracy"]
result_cols1 = ["Classifier","Precision", "Recall", 'F-measure']

result_frame = pd.DataFrame(columns=result_cols)
result_frame1 = pd.DataFrame(columns=result_cols1)

classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        SGDClassifier(),
        LogisticRegression(),
        SVC(),
        GaussianNB(),
        RandomForestClassifier(),
        MLPClassifier()]


type1_error = []
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    acc = accuracy_score(y_test,predicted)
    precision = precision_score(y_test, predicted, average='weighted')
    rec = recall_score(y_test, predicted, average='weighted')
    f_measure = f1_score(y_test, predicted, average='weighted')
    type1_error.append([precision, rec, f_measure])

    print (name+' accuracy = '+str(acc*100)+'%')
    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)
    result_frame = result_frame.append(acc_field)
    
    acc_field1 = pd.DataFrame([[name, precision, rec, f_measure]], columns=result_cols1)
    result_frame1 = result_frame1.append(acc_field1)
    confusion_mc = confusion_matrix(y_test, predicted)
    df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(num)], columns = [i for i in range(num)])
    plt.figure(figsize=(5.5,4))
    sns.heatmap(df_cm, annot=True)
    plt.title(name+'\nAccuracy:{0:.3f}'.format(acc))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
plt.figure()
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

df1 = pd.melt(result_frame1, id_vars=['Classifier']).sort_values(['variable','value'])

plt.figure()
sns.barplot(x="Classifier", y="value", hue="variable", data=df1)
plt.xticks(rotation=90)
plt.show()