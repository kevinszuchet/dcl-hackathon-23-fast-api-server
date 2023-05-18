#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:23:06 2023

@author: nahuelpatino
"""

import numpy as np
import pandas as pd


res=np.loadtxt("simmilarity_matrix.csv", delimiter=";")
ct=np.loadtxt("occurrence_matrix.csv", delimiter=";")
w_data=pd.read_csv('Salesforce_blip-vqa-base2.csv')


# Just in case:
for i in range(len(res)):
    res[i,i]=0
for i in range(len(ct)):
    ct[i,i]=0


def affinity_recommendation(item_id):
    ix = w_data[w_data['ITEM_ID'] == item_id].index.to_numpy()[0]

    drop_threshold_by=0
    l=0
    while l ==0:
        ixs=np.argwhere( res[:,ix] > (0.85 - drop_threshold_by) ).flatten()
        l=len(ixs)
        drop_threshold_by +=0.05
        
    totals = ct[:,ixs].sum(axis=1).flatten()
    x = np.argsort(totals)[::-1][:5]
    rec=w_data.iloc[list(x.astype(int)) ,1].tolist()
    return rec

def aesthetic_simmilarity_recommendation(item_id):
    ix = w_data[w_data['ITEM_ID'] == item_id].index.to_numpy()[0]
    similar_ixs =np.argsort( res[:,ix] )[-5:]
    rec=w_data.iloc[list(similar_ixs.astype(int)) ,1].tolist()
    return rec


affinity_recommendation('0x872ee52a139d3e2c4ec714f9150cb707dd429f5c-0')
aesthetic_simmilarity_recommendation('0x872ee52a139d3e2c4ec714f9150cb707dd429f5c-0')
