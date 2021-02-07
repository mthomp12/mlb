# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 07:23:28 2020

@author: mthom
"""


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from password import password

length = 500

pw = password()
con = create_engine("mysql+mysqlconnector://{0}:{1}@localhost/mlb_db".format(pw.user,pw.password))

sql = '''Select ab.game_id, ab.start, home, ab.id, opp, team, PA, AB, SAC, HBP, BB, S, D, T, HR, TB, H, SO, RBI, R, `batter hand`, `pitcher hand`, `lineup position`, CS, SB, 
fanduel_points as pts from at_bats ab 
LEFT JOIN batter_points bp ON ab.game_id = bp.game_id AND ab.id = bp.id
where year(ab.start)>=2006 and team is not null order by ab.id, ab.start desc, ab.game_id desc, inning desc, batter_num desc'''

df = pd.read_sql(sql, con)

df['stadium'] = np.where( df['home']==1, df['team'], df['opp'])
stadium = pd.get_dummies(df['stadium'], prefix='b')
df[stadium.columns] = stadium


df['opposite hand'] = np.where(df['batter hand']==df['pitcher hand'], 0, 1)

df['yr'] = df['start'].apply(lambda x: x.year)
df['idx'] = range(len(df))

#Determine how many rows batter has from current record
df['maxidx'] = df[['id','idx']].groupby('id').transform('max')
df['remain'] = df['maxidx'] - df['idx']

 #Set target
for col in ['pts','game_id','opposite hand','stadium']:
    df['nxt_{}'.format(col)] = df[['id',col]].groupby('id').shift(1)
      
df['pred_flag'] = np.where((df['yr']>2006)&(df['game_id']!=df['nxt_game_id'])&df['nxt_pts'].notna(), 1, 0)    
df['nxt_opposite hand'] = np.where(df['pred_flag']==1, df['nxt_opposite hand'], 0)

stadium_dict = {'CHA': 0, 'DET': 1, 'MIN': 2, 'SEA': 3, 'CLE': 4, 'ATL': 5, 'ANA': 6, 'PHI': 7, 'TBA': 8, 'KCA': 9, 
                'OAK': 10, 'BOS': 11, 'TEX': 12, 'CHN': 13, 'WAS': 14, 'HOU': 15, 'TOR': 16, 'BAL': 17, 'NYA': 18, 
                'CIN': 19, 'PIT': 20, 'SLN': 21, 'COL': 22, 'NYN': 23, 'MIA': 24, 'MIL': 25, 'SDN': 26, 'SFN': 27, 
                'ARI': 28, 'LAN': 29, 'FLO':30}

#use one-hot encoded vertically
df['nxt_stadium'] = df['nxt_stadium'].map(stadium_dict)
    
df['nxt_opposite hand'] = np.where(df['pred_flag']==1, df['nxt_opposite hand'], 0)
#load pitcher data
p = pd.read_pickle('pitchers.pkl')
p.rename(columns={'team_p':'opp'}, inplace=True)

df['idx'] = range(len(df))
idx = df[(df['pred_flag']==1)][['idx','remain','maxidx','yr']]
idx['end'] = idx['idx'] + length
idx['end'] = idx.apply(lambda df: min(df['end'],df['maxidx']), axis=1)
idx = idx[idx['idx']!=idx['end']]
idx.reset_index(inplace=True, drop=True)


p_cols = [col for col in p.columns if col not in ['game_id','opp','rows_p','id']]    
b_cols = ['home', 'PA', 'AB', 'SAC', 'HBP', 'BB',
       'S', 'D', 'T', 'HR', 'TB', 'H', 'SO', 'RBI', 'R', 
       'CS', 'SB', 'pts','nxt_stadium','opposite hand','nxt_opposite hand'] + stadium.columns.tolist()

agg = df[b_cols].agg(['min','max'])
scale_cols = agg.loc[:,agg.loc['max']!=1].columns.tolist()
scale_cols.remove('nxt_stadium')
df[scale_cols] = (df[scale_cols] - df.iloc[0:50000][scale_cols].mean())/df.iloc[0:50000][scale_cols].std()


null_ct = []
pnull = []
pitcher_ct = 0
train = idx[idx['yr']<2016].index.tolist()
for i in train[0:90000]:
    x = df.iloc[idx.iloc[i]['idx']:idx.iloc[i]['end']].copy()
    x.reset_index(inplace=True, drop=True)
    x.loc[1:, 'nxt_opposite hand'] = 0
    
    #vertical one-hot encoding next stadium
    stad = int(x.iloc[0]['nxt_stadium'])
    x['nxt_stadium'] = 0
    x.loc[stad, 'nxt_stadium'] = stad
    if x[b_cols+['nxt_pts']].isna().sum().sum():
        null_ct += [x[b_cols+['nxt_pts']].isna().sum().max()]
        x = x.dropna(subset=b_cols+['nxt_pts']).copy()
        x.reset_index(inplace=True, drop=True)
    while len(x) < 500:
        x.loc[len(x)] = 0
    if x[b_cols+['nxt_pts']].isna().sum().sum():
        raise Exception()
    if len(p[(p['game_id']==x.iloc[0]['game_id'])&(p['opp']==x.iloc[0]['opp'])]):
        pidx = p[(p['game_id']==x.iloc[0]['game_id'])&(p['opp']==x.iloc[0]['opp'])].index[0]
        pend = pidx + p.loc[pidx, 'rows_p'] - 1
        p0 = p.loc[pidx:pend, p_cols]
        pitcher_ct += 1
        if p0.isna().sum().sum():
            pnull+= [p0.isna().sum().max()]
            p0.dropna(inplace=True)
        p0 = np.array(p0)
        fill = np.zeros((length - p0.shape[0], len(p_cols)))
        p0 = np.concatenate((p0,fill))
    else:
        p0 = np.zeros((length, len(p_cols)))
    x0 = np.array(x[b_cols])
    x1 = np.array(x['nxt_pts']).reshape(-1,1)
    x_all = np.concatenate((x0, p0, x1), axis=1)
    np.save('./data/train_x/x'+str(i-train[0])+'.npy', x_all)
    
if null_ct:
    null_train = pd.Series(null_ct).value_counts()
if pnull:
    null_train_p = pd.Series(pnull).value_counts()



xnull_ct = []
pnull = []

cv = idx[idx['yr']==2016].index.tolist()
for i in cv[0:10000]:
    x = df.iloc[idx.iloc[i]['idx']:idx.iloc[i]['end']].copy()
    x.reset_index(inplace=True, drop=True)
    x.loc[1:, 'nxt_opposite hand'] = 0
    
    #vertical one-hot encoding next stadium
    stad = int(x.iloc[0]['nxt_stadium'])
    x['nxt_stadium'] = 0
    x.loc[stad, 'nxt_stadium'] = stad
    if x[b_cols+['nxt_pts']].isna().sum().sum():
        null_ct += [x[b_cols+['nxt_pts']].isna().sum().max()]
        x = x.dropna(subset=b_cols+['nxt_pts']).copy()
        x.reset_index(inplace=True, drop=True)
    while len(x) < 500:
       x.loc[len(x)] = 0
    if x[b_cols+['nxt_pts']].isna().sum().sum():
        raise Exception()
    if len(p[(p['game_id']==x.iloc[0]['game_id'])&(p['opp']==x.iloc[0]['opp'])]):
        pidx = p[(p['game_id']==x.iloc[0]['game_id'])&(p['opp']==x.iloc[0]['opp'])].index[0]
        pend = pidx + p.loc[pidx, 'rows_p'] - 1
        p0 = p.loc[pidx:pend, p_cols]
        if p0.isna().sum().sum():
            pnull+= [p0.isna().sum().max()]
            p0.dropna(inplace=True)
        p0 = np.array(p0)
        fill = np.zeros((length - p0.shape[0], len(p_cols)))
        p0 = np.concatenate((p0,fill))
    else:
        p0 = np.zeros((length, len(p_cols)))
    x0 = np.array(x[b_cols])
    x1 = np.array(x['nxt_pts']).reshape(-1,1)
    x_all = np.concatenate((x0, p0, x1), axis=1)
    np.save('./data/cv_x/x'+str(i-cv[0])+'.npy', x_all)
    
if null_ct:    
    null_cv = pd.Series(null_ct).value_counts()
if pnull:
    null_cv_p = pd.Series(pnull).value_counts()




xnull_ct = []
pnull = []
pitcher_ct = 0
test = idx[idx['yr']>2016].index.tolist()
for i in test[1000:]:
    x = df.iloc[idx.iloc[i]['idx']:idx.iloc[i]['end']].copy()
    x.reset_index(inplace=True, drop=True)
    x.loc[1:, 'nxt_opposite hand'] = 0
    
    #vertical one-hot encoding next stadium
    stad = int(x.iloc[0]['nxt_stadium'])
    x['nxt_stadium'] = 0
    x.loc[stad, 'nxt_stadium'] = stad
    if x[b_cols+['nxt_pts']].isna().sum().sum():
        null_ct += [x[b_cols+['nxt_pts']].isna().sum().max()]
        x = x.dropna(subset=b_cols+['nxt_pts']).copy()
        x.reset_index(inplace=True, drop=True)
    while len(x) < 500:
       x.loc[len(x)] = 0
    if x[b_cols+['nxt_pts']].isna().sum().sum():
        raise Exception()
    if len(p[(p['game_id']==x.iloc[0]['game_id'])&(p['opp']==x.iloc[0]['opp'])]):
        pitcher_ct += 1
        pidx = p[(p['game_id']==x.iloc[0]['game_id'])&(p['opp']==x.iloc[0]['opp'])].index[0]
        pend = pidx + p.loc[pidx, 'rows_p'] - 1
        p0 = p.loc[pidx:pend, p_cols]
        if p0.isna().sum().sum():
            pnull+= [p0.isna().sum().max()]
            p0.dropna(inplace=True)
        p0 = np.array(p0)
        fill = np.zeros((length - p0.shape[0], len(p_cols)))
        p0 = np.concatenate((p0,fill))
    else:
        p0 = np.zeros((length, len(p_cols)))
    x0 = np.array(x[b_cols])
    x1 = np.array(x['nxt_pts']).reshape(-1,1)
    x_all = np.concatenate((x0, p0, x1), axis=1)
    np.save('../NN/data/test_x/x_{0}_{1}.npy'.format(x.loc[0]['id'], x.loc[0]['nxt_game_id']), x_all)
    
if null_ct:    
    null_test = pd.Series(null_ct).value_counts()
if pnull:
    null_test_p = pd.Series(pnull).value_counts()
