#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:39:58 2018

@author: HM
"""

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
import CSVDataManagement as dmanager
from triedpy import triedacp as tacp

#%% Import data
print('Récupération des données')

data = dmanager.get_all_features("dataPresqueSansNan")

#%% Get Training/Test Sets
print("Mise en forme des données")
df = []
for k in range(len(data)):
    df += [data[k].loc[data[k]['pft'].dropna().index]]

feature = ['chl', 'sst' ,'l555','l490', 'l443','l412','lat','lon','date']
#feature = ['l555','l490', 'l443','l412']

X = pd.DataFrame(df[0][feature],columns=feature)
y = pd.DataFrame(df[0]['pft'],columns=['pft'])
#y = pd.DataFrame(df[0]['chl'],columns=['chl'])

for k in range(1,len(data)):
    X = X.append(pd.DataFrame(df[k][feature]), ignore_index=True)
    y = y.append(pd.DataFrame(df[k]['pft']), ignore_index=True)

print("Preprocessing")

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=324)

#%% Train Classifier
"""
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20), random_state=1)
clf.fit(X_train, y_train)

predictions = clf.predict(X_valid)
print(mean_squared_error(y_true = y_valid, y_pred = predictions))

"""
#%% Print ACP

def print_acp_from_data(Xtrain,ytrain,title):
    Xprepro = preprocessing.StandardScaler().fit_transform(Xtrain)
    acp = PCA(n_components=2)
    acp.fit(Xprepro, ytrain)
    X = acp.transform(Xprepro)
    print(len(X))
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

    tacp.corcer(Xprepro,X,1,2,varnames=feature)
    plt.show()

    return X, acp

    
transfo, pca = print_acp_from_data(X_train,y_train,"le phytoplancton")

#%% Print cercle corrélation
print("Cercle de corrélation en cours")
Xprepro = preprocessing.StandardScaler().fit_transform(X_train)

X=Xprepro
y=y_train

pca=PCA()
pca.fit(X,y)
x_new=pca.transform(X)


def myplot(score,coeff,labels=feature):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    a = 0
    b = 1
    #plt.scatter(xs ,ys, c = y) #without scaling
    for i in range(n):
        plt.arrow(0, 0, coeff[i,a], coeff[i,b],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,a]* 1.15, coeff[i,b] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,a]* 1.15, coeff[i,b] * 1.15, labels[i], color = 'r', ha = 'center', va = 'center')
            plt.xlim(-1.2,1.2)
            plt.ylim(-1.2,1.2)

            
Xprepro = preprocessing.StandardScaler().fit_transform(x_new)
myplot(Xprepro[:,0:2],pca.components_)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()

print("Variance expliquée")
print(pca.explained_variance_ratio_)
