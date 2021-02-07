# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:45:27 2020

@author: mthom
"""

import pandas as pd

df1000 = pd.read_pickle('./ml_data 1000 2021.pkl')

df100 = pd.read_pickle('./ml_data 100 2021.pkl')
df100 = df100[['pts_100']]

df250 = pd.read_pickle('./ml_data 250 2021.pkl')
df250 = df250[['pts_250']]

df500 = pd.read_pickle('./ml_data 500 2021.pkl')
df500 = df500[['pts_500']]

df2000 = pd.read_pickle('./ml_data 2000 2021.pkl')
df2000 = df2000[['pts_2000']]
#get ratio_x = ts_x_parkadji/pts_1000_parkadj for 500, 250 and 100 


#merge dfs
df = pd.concat([df1000, df100, df250, df500, df2000], axis = 1)

#df.shape
for i in [100, 250, 500, 2000]:
  df['ratio_'+str(i)] = df['pts_'+str(i)]/df['pts_1000']


df1 = df.dropna()
df1.to_pickle('combined 2021 fixed.pkl')

