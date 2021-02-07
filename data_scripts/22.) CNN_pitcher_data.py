# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:32:06 2020

@author: mthom
"""


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from password import password

length = 500

pw = password()
con = create_engine("mysql+mysqlconnector://{0}:{1}@localhost/mlb_db".format(pw.user,pw.password))

sql = '''Select ab.game_id, ab.start, team as opp, ab.P as id, opp as team, 1-home as home, PA, AB, SAC, HBP, BB, S, D, T, HR, TB, H, SO, RBI, R, CS, SB, 
fanduel_points as pts from at_bats ab 
LEFT JOIN batter_points bp ON ab.game_id = bp.game_id AND ab.id = bp.id
where year(ab.start)>=2010 order by ab.id, ab.start desc, ab.game_id desc, inning desc, batter_num desc'''

df = pd.read_sql(sql, con)

sql = '''SELECT game_id, pitcher1 as id FROM games WHERE DATE/10000>2015
UNION
SELECT game_id, pitcher0 as id FROM games WHERE DATE/10000>2015'''

starters = pd.read_sql(sql, con)
starters['starter'] = 1

df = df.merge(starters, how='left', on=['game_id','id'], validate='m:1')
df = df[df['starter']==1]
df.drop('starter', inplace=True, axis=1)
df.reset_index(inplace=True, drop=True)

df['stadium'] = np.where( df['home']==0, df['team'], df['opp'])



grp = df.groupby(['game_id','id']).sum().reset_index()
grp['home'] = grp['home'].apply(lambda x: min(1, x))


sql = 'Select id, team, date, IP, H, R, ER, BB, SO, W, L, QS, fanduel_points as pts_p, number from pitcher_points'
pitcher = pd.read_sql(sql, con)


df['idx'] = range(len(df))
df['minidx'] = df[['game_id','id','idx']].groupby(['game_id','id']).transform('min')
single = df[df['idx']==df['minidx']][['game_id','id','stadium','start','team']]

single['date'] = single['start'].apply(lambda x: x.date())
single['number'] = single['game_id'].apply(lambda x: x[-1])
single['number'] = single['number'].astype('int')

grp = grp.merge(single, how='left', on=['game_id','id'], validate='m:1')

pitcher.drop(['BB','H','SO','R'], inplace=True, axis=1)
grp = grp.merge(pitcher, how='left', on=['date','id','team','number'], validate='m:1')

grp['flag'] = 1
pitcher = pitcher.merge(grp[grp['IP'].isna()][['date','id','team','flag']], how='left', on=['date','id','team'], validate='m:1')
pitcher = pitcher[pitcher['flag']==1]
pitcher.drop(['flag','number'], inplace=True, axis=1)


lenCheck = len(grp)
grp_a = grp[grp['IP'].notna()]
grp_b = grp[grp['IP'].isna()]
grp_b.drop(['IP', 'ER', 'W', 'L', 'QS', 'pts_p'], axis=1, inplace=True)
grp_b = grp_b.merge(pitcher, how='left', on=['date','id','team'], validate='m:1')
grp = pd.concat([grp_a,grp_b])

if len(grp_a)+len(grp_b)-len(grp):
    raise Exception('Combine Error')


grp.drop('start', axis=1, inplace=True)
cols = [col for col in grp.columns if col not in ['game_id','id','pts','pts_p','date','number']]
grp.rename(columns=dict(zip(cols,[col+'_p' for col in cols])), inplace=True)

grp.sort_values(['id','date'], ascending=[True,False], inplace=True)
grp.rename(columns={'pts':'pts_against_pitcher'}, inplace=True)


locs = pd.get_dummies(grp['stadium_p'], prefix='p')
grp[locs.columns] = locs
grp.drop(['stadium_p','date','number'], inplace=True, axis=1)

grp.reset_index(inplace=True, drop=True)
grp['idx'] = range(len(grp))
grp['maxidx'] = grp[['id','idx']].groupby('id').transform('max')
grp['rows_p'] = grp['maxidx'] - grp['idx'] + 1
grp.drop(['idx','maxidx'], inplace=True, axis=1)
grp.to_pickle('pitchers.pkl')

