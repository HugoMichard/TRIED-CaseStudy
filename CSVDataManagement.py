#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:56:40 2018

@author: HM
"""

import pandas as pd


"""     Récupération des données fournies     """
def get_features_from_date(directory, date):
    
    directory = directory + '/DonneesData'
    df1 = pd.read_csv(directory+str(date)+'.csv',index_col=0)
    return df1

def get_all_features(directory, limit=None):
    var = []
    if limit is None:
        for k in range(184):
            #print(list(dataset.variables.keys()))
            #print(dataset.variables["PFT"])
    
            var += [get_features_from_date(directory,k)]
    else:
        for k in range(limit):
            var += [get_features_from_date(directory,k)]
    return var
