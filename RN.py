#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:39:58 2018

@author: HM
"""

import CSVDataManagement as dmanager
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%% Import data

data = dmanager.get_all_features("dataPresqueSansNan")

#%% Get Training/Test Sets
df = []
for k in range(len(data)):
    df += [data[k].loc[data[k]['pft'].dropna().index]]


feature = ['chl', 'sst' ,'l555','l490', 'l443','l412','lat','lon','date']


X =  pd.DataFrame(df[0][feature],columns=feature)
y = pd.DataFrame(df[0]['pft'],columns=['pft'])
for k in range(1,len(data)):
    X.append(pd.DataFrame(df[k][feature]))
    y.append(pd.DataFrame(df[k]['pft']))

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=324)


#%% Train Classifier
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20), random_state=1)
clf.fit(X_train, y_train)

predictions = clf.predict(X_valid)
print(mean_squared_error(y_true = y_valid, y_pred = predictions))


#%% Print ACP
def print_acp_from_data(Xtrain,ytrain,title):
    X = preprocessing.StandardScaler().fit_transform(Xtrain)
    acp = PCA(n_components=3)
    acp.fit(X)
    X = acp.transform(X)
    ytrain = ytrain.values.flatten()
    legends = []
    for k in range(1,8):
        elements = np.where((ytrain.astype(int))==k)
        legends += [k]

        Xcurrent = X[elements]

        plt.scatter(Xcurrent[:,0],Xcurrent[:,1],s=0.5)
    plt.legend(legends)
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    eigen = np.sum(acp.explained_variance_ratio_)*100
    plt.title("ACP pour "+title+", variance expliquée de : " + str(round(eigen,2)))
    plt.show()
    
print_acp_from_data(X_train,y_train,"le phytoplancton")