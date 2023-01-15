# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import timedelta

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32', 'onpromotion':'bool'}

print('-------------------------- LOADING --------------------------')
train = pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv', usecols=[1,2,3,4,5], dtype=dtypes, parse_dates=['date']
                    ,skiprows=range(1, 86672217) #Skip dates before 2016-08-01
                    )
test = pd.read_csv('../input/favorita-grocery-sales-forecasting/test.csv', parse_dates=['date'])
items = pd.read_csv('../input/favorita-grocery-sales-forecasting/items.csv')


#testList=['96995','99197','103501','103520','103665','105574',
#'105575','105576','105577','105693','105737','105857','106716',
#'108079','108634','108696','108698','108701','108786','108797']
#train = train[-(train['item_nbr'].isin(testList))]



print('-------------------------- LOADED, MERGING ITEM DATA --------------------------')
test = pd.merge(test, items[['item_nbr','class']], how='left', on=['item_nbr'])
test.loc[:, 'class'].fillna(0, inplace=True)

print('-------------------------- LOADED, CONVERTING DATA --------------------------')
train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives

print('-------------------------- CONVERTED, CALCULATING DAILY/WEEKLY MEANS --------------------------')
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek
#Days of Week Means
#By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(
        ['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw').reset_index()  
class_ma_dw = pd.merge(ma_dw, items[['item_nbr','class']], how='left', on=['item_nbr'])
class_ma_dw.loc[:, 'class'].fillna(0, inplace=True)
class_ma_dw = class_ma_dw.groupby(['dow','store_nbr','class'])['madw'].mean().to_frame('class_madw').reset_index()

ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(
        ['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk').reset_index()
class_ma_wk = pd.merge(ma_wk, items[['item_nbr','class']], how='left', on=['item_nbr'])
class_ma_wk.loc[:, 'class'].fillna(0, inplace=True)       
class_ma_wk = class_ma_wk.groupby(['store_nbr','class'])['mawk'].mean().to_frame('class_mawk').reset_index()

train.drop('dow',1,inplace=True)

print('-------------------------- CALCULATED, CREATING MISSING RECORDS --------------------------')
# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
).reset_index()

del u_dates, u_stores, u_items

train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs
lastdate = train.iloc[train.shape[0]-1].date

print('-------------------------- CREATED, CALCULATING MOVING AVERAGES --------------------------')
#Moving Averages
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(
        ['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais')

for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')

del tmp,tmpg,train

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)
ma_is.drop(list(ma_is.columns.values)[3:],1,inplace=True)


class_ma_is = pd.merge(ma_is, items[['item_nbr','class']], how='left', on=['item_nbr'])
class_ma_is.loc[:, 'class'].fillna(0, inplace=True)       
class_ma_is = class_ma_is.groupby(['store_nbr','class'])['mais'].mean().to_frame('class_mais').reset_index()
#flatten out any outliers
mean = np.mean(class_ma_is['class_mais'], axis=0)
sd = np.std(class_ma_is['class_mais'], axis=0)
lowerBound = mean - 2 * sd
upperBound = mean + 2 * sd

class_ma_is['class_mais'] = class_ma_is['class_mais'].apply(lambda mais : lowerBound if mais<lowerBound else mais)
class_ma_is['class_mais'] = class_ma_is['class_mais'].apply(lambda mais : upperBound if mais>upperBound else mais)



#Load test
print('-------------------------- CALCULATED, LOADING TEST --------------------------')
test['dow'] = test['date'].dt.dayofweek
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])
test = pd.merge(test, class_ma_is, how='left', on=['class','store_nbr'])
test = pd.merge(test, class_ma_wk, how='left', on=['class','store_nbr'])
test = pd.merge(test, class_ma_dw, how='left', on=['class','store_nbr','dow'])

del ma_is, ma_wk, ma_dw

#Forecasting Test
print('-------------------------- LOADED, FORECASTING --------------------------')
test['unit_sales'] = test.mais 
pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'unit_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']

#Now do classes for instances where mais has nothing for that specific item
class_idx = (test['mais'].isnull())
#| (test['mais'] == 0)
test_class_pos = test.loc[class_idx]
test.loc[class_idx, 'unit_sales'] = test_class_pos['class_mais'] * 0.5 * test_class_pos['class_madw'] / test_class_pos['class_mawk']

#fill remainings with 0 (should there be any?)
test.loc[:, "unit_sales"].fillna(0, inplace=True)

test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values 
#apply promotion multiplier when relevant
test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.5




print('-------------------------- DONE, EXPORTING --------------------------')
test[['id','unit_sales']].to_csv('submission.csv.gz', index=False, float_format='%.3f', compression='gzip')