#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:30:14 2018

@author: HM
"""

import itertools
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

def merge_array(data):
    """
        
    merged = data[0].tolist()
    print(type(merged))
    for l in data:
        merged.append(l.tolist())
    """
    merged = list(itertools.chain.from_iterable(data))
    return merged
    

def get_index_all_nulls(data):
    nulls = []
    for k in range(len(data)):
        temp = np.argwhere(np.isnan(data[k]))
        temp = temp[:,0]
        nulls += [temp]
    nulls = merge_array(nulls)
    print(nulls)
    return list(set(nulls))


    
def concatenate_index(index1, index2):
    """ concatenate 2 arrays and all values are unique"""

"""
def remove_all_index_from_data(data, index):
    for k in range(len(index)):
        if(np.delete())
        """
def put_array_in_shape(dataset, dim3=False):
    dataset = np.array(dataset)
    print(dataset.shape)
    if(dim3):
        nsamples,nx,ny = dataset.shape
        return dataset.reshape((nsamples,nx*ny))
    else: #In dimension 4
        nsamples,variables,nx,ny = dataset.shape
        return dataset.reshape((nsamples,variables*nx*ny))
    
sst, lat, lon = get_all_data_from("SST", "SST", True)

"""
chl = get_all_data_from("Chl", "CHL-OC5_mean")
pft = get_all_data_from("PFT", "PFT")
l555 = get_all_data_from("555", "NRRS555_mean")
l490 = get_all_data_from("490", "NRRS490_mean")
l443 = get_all_data_from("443", "NRRS443_mean")
l412 = get_all_data_from("412", "NRRS412_mean")
"""

print(type(sst))
ssts = put_array_in_shape(sst,dim3=True)
chl_nulls = get_index_all_nulls(ssts)
print(chl_nulls)

"""
X = []
y = []
print(len(pft))
for k in range(len(pft)):
    X += [np.array([chl[k],sst[k],l555[k],l490[k],l443[k],l412[k]])]
    y += [pft[k]]
"""
"""
a= np.array([[1,2],[2,3]])
test = np.unique(chl_nulls[0])
test2 = np.unique(chl_nulls[1])
print(np.unique([test,test2]))
"""

def 