#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:15:18 2018

@author: HM et 
"""
#test

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset
from sklearn.neighbors import KNeighborsClassifier as knc
import random as rnd
from sklearn.model_selection import train_test_split

"""     Variables dependant de l'environnement    """
data_directory_path = "donnees/"



"""             """

"""     Récupération des données fournies     """
def get_all_data_from(variable_path, variable, get_coord=False):
    directory = data_directory_path + variable_path
    allfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    
    var = []
    for k in allfiles:
        dataset = Dataset(directory + '/' + k)
        #print(list(dataset.variables.keys()))
        #print(dataset.variables["PFT"])

        var += [dataset.variables[variable][:,:]]

    if(get_coord):
        lon = dataset.variables['lon'][:]
        lat = dataset.variables['lat'][:]
        var = sp.asarray(var)
        lat = sp.asarray(lat)
        lon = sp.asarray(lon)
        return var, lat, lon

    else:
        var = sp.asarray(var)
        return var


chl = get_all_data_from("Chl", "CHL-OC5_mean")
sst, lat, lon = get_all_data_from("SST", "SST", True)
pft = get_all_data_from("PFT", "PFT")
l555 = get_all_data_from("555", "NRRS555_mean")
l490 = get_all_data_from("490", "NRRS490_mean")
l443 = get_all_data_from("443", "NRRS443_mean")
l412 = get_all_data_from("412", "NRRS412_mean")

X = []
y = []
print(len(pft))
for k in range(len(pft)):
    X += [np.array([chl[k],sst[k],l555[k],l490[k],l443[k],l412[k]])]
    y += [pft[k]]


def put_array_in_shape(dataset, dim3=False):
    dataset = np.array(dataset)
    print(dataset.shape)
    if(dim3):
        nsamples,nx,ny = dataset.shape
        return dataset.reshape((nsamples,nx*ny))
    else:
        nsamples,variables,nx,ny = dataset.shape
        return dataset.reshape((nsamples,variables*nx*ny))

    
    
X = put_array_in_shape(X)
y = put_array_in_shape(y, True)

Xtrain, ytrain, Xtest, ytest = train_test_split(X,y,test_size=0.33)
#Xtrain, ytrain, Xcv, ycv = train_test_split(Xtrain,ytrain,test_size=0.5)

classifier_tp = knc(n_neighbors = 1)
classifier_tp.fit(Xtrain,ytrain)




"""
plt.figure(2)
#plt.matshow(sst[0],cmap='Paired')
plt.matshow(sst[0])
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

plt.figure(3)
plt.matshow(np.nanmean(sst, axis=0))
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

plt.figure(4)
plt.matshow(np.nanstd(sst, axis=0))
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


moy1 = np.nanmean(sst, axis=1)
moy_tot = np.nanmean(moy1, axis=1)
plt.figure(5)
plt.plot(range(184),moy_tot,'-o')
plt.title("SST moyen méditerranéen par intervalle de 8 jours")
plt.xlabel("Nombre d'intervalles depuis le 01/01/2014")
plt.ylabel("Température en °C")
plt.show()
"""
