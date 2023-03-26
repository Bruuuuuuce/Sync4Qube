import os
import sys
import time
import MySQLdb
import pandas as pd
import numpy as np
import seaborn as sns

import statsmodels.api as sm
import multiprocessing as mp
import matplotlib.pyplot as plt
from PqiDataSdk import *
from PqiData import *
import multiprocessing as mp
import statsmodels.api as sm
sys.path.append('/home/zywang/20 factor/01 manual_factor/tools/')
from backtest_cdb import backtest_bycdb
    
ds = PqiDataSdk(user="zywang", size=16, pool_type="mt")

# 常用数据区
end_date = ds.get_eod_history()['ClosePrice'].columns[-1]
eod_data = ds.get_eod_history(start_date='20150101',
                              end_date=end_date,
                              price_mode='AFTER')
twap_open = ds.get_eod_history(start_date='20150101',
                               end_date=end_date,
                               source='ext_stock',
                               price_mode='AFTER')['TwapBegin60']
rtn_df = twap_open.shift(-2,axis=1)/twap_open.shift(-1,axis=1)-1
tickers = list(eod_data['ClosePrice'].index)
style_data = ds.get_factor_exposure(start_date='20150101',
                              end_date=end_date,
                              tickers=tickers)

o = eod_data['OpenPrice']
c = eod_data['ClosePrice']
h = eod_data['HighestPrice']
l = eod_data['LowestPrice']
v = eod_data['VWAP']
tval = eod_data['TradeValue']
tovr = eod_data['TurnoverRate']
size = style_data['size']
beta = style_data['beta']
bp = style_data['book_to_price']

twap_o = twap_open
open_lim = (eod_data['UpLimitPrice']==eod_data['OpenPrice']).astype('int')



mom1 = c/c.shift(1,axis=1) - 1


ind_df = ds.get_sw_members(level=1)
kwd_list = ['银行','非银金融','房地产','钢铁','公用事业']
mask_list = []
for kwd in kwd_list:
    mask_list.extend(list(ind_df[ind_df['industry_name']==kwd]['con_code']))

st_mask = (1-eod_data['STStatus'].replace(9,1))
st_mask = st_mask/st_mask


def get_idx_mask(ticker):
    mask = ds.get_index_weight(ticker=ticker,
                            start_date='20160101',
                            end_date=end_date,
                            format='eod')
    return mask/mask
zz300_mask = get_idx_mask('000300')
zz500_mask = get_idx_mask('000905')
zz1000_mask = get_idx_mask('000852')
all_mask = zz1000_mask.fillna(1)
kcb_mask = all_mask.replace(1,np.nan)
kcb_mask.loc[[x for x in kcb_mask.index if x.startswith('68')]] = 1
cyb_mask = all_mask.replace(1,np.nan)
cyb_mask.loc[[x for x in cyb_mask.index if x.startswith('30')]] = 1
others_mask = ((all_mask - zz300_mask.fillna(0)\
                         -zz500_mask.fillna(0)\
                         -zz1000_mask.fillna(0)\
                         -kcb_mask.fillna(0)\
                         -cyb_mask.fillna(0))==1).astype('int')
non_1800_mask = ((all_mask - zz300_mask.fillna(0)\
                         -zz500_mask.fillna(0)\
                         -zz1000_mask.fillna(0))==1).astype('int')
zz1800_mask = ((zz300_mask.fillna(0)+zz500_mask.fillna(0)+zz1000_mask.fillna(0))==1).astype('int')

mask_dict = {}
mask_dict['300'] = zz300_mask.replace(0,np.nan)
mask_dict['500'] = zz500_mask.replace(0,np.nan)
mask_dict['1000'] = zz1000_mask.replace(0,np.nan)
mask_dict['cyb'] = cyb_mask.replace(0,np.nan)
mask_dict['kcb'] = kcb_mask.replace(0,np.nan)
mask_dict['others'] = others_mask.replace(0,np.nan)
mask_dict['all'] = all_mask.replace(0,np.nan)
mask_dict['non_1800'] = non_1800_mask.replace(0,np.nan)
mask_dict['1800'] = zz1800_mask.replace(0,np.nan)
# 当日涨跌停剔除
mask_uplim = (eod_data['UpLimitPrice']/eod_data['HighestPrice']-1<0.005)
mask_downlim = (eod_data['LowestPrice']/eod_data['DownLimitPrice']-1<0.005)
mask_lim = mask_uplim | mask_downlim
# 相关性检验
period_list = [1,3,5,10,20]
mom = c/c.shift(1,axis=1)
rr = (h-l)/(h+l)
pv_dict = {}
for p in period_list:
    pv_dict['mom_{}'.format(p)] = mom.rolling(p,axis=1,min_periods=1).mean()
    pv_dict['rr_{}'.format(p)] = rr.rolling(p,axis=1,min_periods=1).mean()
    pv_dict['tovr_{}'.format(p)] = tovr.rolling(p,axis=1,min_periods=1).mean()
pv_dict['on'] = c/o - 1 
pv_dict['vc'] = v/c - 1
pv_dict['size'] = eod_data['FloatMarketValue']
pv_dict['tval'] = eod_data['TradeValue']
# 工具函数区

# 读取工具
def get_fac(fac_name,fac_path):
    fac = ds.get_eod_feature(fields=[fac_name],where=fac_path)[fac_name]
    fac = (fac.to_dataframe()*mask_dict['1800']).dropna(how='all',axis=1)
    return fac

def get_con(table_name,fac_name):
    data = ds.get_forecast(table=table_name,fields=['stock_code','con_date',fac_name])
    data = data.set_index(['stock_code','con_date'])
    raw = data[fac_name].unstack()
    raw.index = list(raw.index)
    raw.columns = [str(x)[:4]+str(x)[5:7]+str(x)[8:10] for x in raw.columns]
    return raw

def get_con_sql(table_name,fac_name):
    conn = MySQLdb.connect("192.168.1.92", "user1", "Gl@202111",'zyyx_ys',port=3306,charset='utf8')
    ticker_df = ds.get_ticker_basic(source='stock')['name']

    sql = "select * from {}".format(table_name) 
    df=pd.read_sql(sql,conn)

    df = df[df.con_date==df.entrytime]
    rrn75 = df[['stock_code','entrytime','relative_report_num_75d']]
    rrn75['entrytime'] = [''.join(str(t).split(' ')[0].split('-')) for t in rrn75['entrytime']]

    test = rrn75.set_index(['stock_code','entrytime'])[fac_name].unstack()
    test.index = list(test.index)
    test.columns = list(test.columns)

    idx = sorted(list(set(test.index)&set(ticker_df.index)))
    test = test.loc[idx]
    return test

def get_fund(fac_name):
    fac = ds.get_eod_history(start_date='20160101',
                          end_date='20220510',
                          fields=[fac_name],
                          source='fundamental')[fac_name]
    return fac

def save_fac(fac_name,fac,fac_path):
    ds.save_eod_feature(where=fac_path,
                        data={fac_name:fac},
                        feature_type='eod',
                        encrypt=False)

def get_pv_corr(fac,method='pearson'):
    corr_dict = {}
    for k in pv_dict.keys():
        corr_dict[k]=fac.corrwith(pv_dict[k],method=method).mean()
    return pd.Series(corr_dict).sort_values(ascending=False)

def test(fac,rtn='TwapOpen2TwapOpen',alpha=True, group=True, ic=True,plot=True,IS=True):
    if IS:
        fac = fac.loc[:,'20180101':'20210331']
    cfg.rtn_type = rtn
    cfg.start_date = fac.columns[0]
    cfg.end_date = fac.columns[-1]
    return backtest_bycdb(fac[(mask_dict['1800']==1)\
                                &(~mask_lim)].rank(pct=True), 
                          cfg, alpha=alpha, group=group, ic=ic,plot=plot)

# 因子处理工具
def neu(fac,styles):
    mask_list_1 = list(set(mask_list)&set(list(fac.index)))
    fac.loc[mask_list_1] = np.nan
    # fac = fac * st_mask
    fac = fac.dropna(how='all',axis=1)
    resid_dict = {}
    for date in fac.columns:
        data = pd.DataFrame()
        data[0] = fac[date]
        for i in range(len(styles)):
            data[i+1] = styles[i][date]
        data = data.dropna()
        idx = data.index
        Y = data.values[:,0]
        X = data.values[:,1:]
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        res = model.fit()
        resid = res.resid
        resid_dict[date] = pd.Series(resid,index=idx)
    resid_df = pd.DataFrame(resid_dict)
    return resid_df

def ind_neu(fac,level=1):
    ind_df = ds.get_sw_members(level=level).set_index('con_code')
    ind_df = ind_df[ind_df['out_date'].isna()]['index_code']
    res = {}
    for date in fac.columns:
        y = fac[date]
        data = pd.concat([y,ind_df],axis=1)
        if data.count()[date]==0:
            res[date] = data[date]
        else:
            data = data.dropna()
            res[date] = data.groupby('index_code').apply(lambda x:x.rank(pct=True))[date]
    res = pd.DataFrame(res)
    return res


def get_ls_sharpe(fac):
    fac = fac.dropna(how='all',axis=1)
    if fac.count().sum()==0:
        return np.nan
    std = fac.rank(pct=True) - 0.5
    std[std>0] = std[std>0]/std[std>0].sum()
    std[std<0] = -std[std<0]/std[std<0].sum()
    ls_ret = (std * rtn_df).sum()
    return ls_ret.mean()/ls_ret.std() * np.sqrt(252)
    
def gini(k:pd.DataFrame):
    if k.min()<0:
        raise TypeError('Gini only support non-negative series')
    k = k.sort_values()/k.sum()
    return 1 - k.sort_values().cumsum().sum()/len(k)

# 画图工具
# heatmap
# wormmap
# double sort
# reg_plot

class Config:
    user = 'zywang'
    tickers = 'all'
    earliest_date = '20160101'
    start_date = "20180101"
    end_date = "20210331"
    indexs = ['000852', '000905']
    benchmark_index = 'zz1800_mean'
    fee_rate = 0.0013
    rtn_type = 'TwapOpen2TwapOpen'
    group_num = 20
    mode = 'exist_df'
    long_short_type = 400
    weight_type = 'factor_value'
    size = 1
cfg = Config()