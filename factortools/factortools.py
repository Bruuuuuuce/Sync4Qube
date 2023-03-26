import datetime
import multiprocessing as mp
import os
import random
import re
import warnings
from functools import partial
from pprint import pprint
import matplotlib.patches as mpathes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from matplotlib import ticker
from numba import jit
from pandarallel import pandarallel
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

def format_factor(df_factor):
    try:
        df_factor.columns = df_factor.columns.map(
            lambda x: x.strftime('%Y%m%d'))
    except:
        pass
    df_factor.columns.name = None
    df_factor.index.name = None


def calc_ic(df_factor, df_ret_eod, plot=False):
    # format_factor(df_factor)
    df_ic = df_factor.corrwith(df_ret_eod)
    df_rank_ic = df_factor.corrwith(df_ret_eod, method='spearman')
    if plot:
        plt.figure(figsize=(20, 5))
        df_ic.plot(label='IC')
        df_rank_ic.plot(label='rank IC')
        plt.grid()
        plt.legend()
    return (df_ic.mean(), df_rank_ic.mean())


def save_factor(name,
                df_factor,
                path='/home/zyding/factor_zyding/eod_datayes',
                feature_type='eod',
                save_method='update'):
    ds.save_eod_feature(data={name: df_factor},
                        where=path,
                        feature_type=feature_type,
                        save_method=save_method,
                        encrypt=False)
    
    
def process_ret(ret_path, window=1, mins=30):
    df = pd.read_parquet(ret_path + f'hfRet_{window}d_{mins}min.parquet')

    df_ret_gp = df.reset_index().groupby(['level_1', 'level_2'])['ret']

    df_ret_rank = df - df
    df_ret_rank['ret'] = df_ret_gp.rank(pct=True).values

    df_ret_alpha = df - df
    df_ret_alpha['ret'] = df_ret_gp.apply(lambda x: x-x.mean()).values

    df_ret_csnorm = df - df
    df_ret_csnorm['ret'] = df_ret_gp.apply(lambda x: (x-x.mean())/(x.quantile(0.9)-x.quantile(0.1))).values

    df_ret_rank.to_parquet(ret_path + f'hfRet_rank_{window}d_{mins}min.parquet')
    df_ret_alpha.to_parquet(ret_path + f'hfRet_alpha_{window}d_{mins}min.parquet')
    df_ret_csnorm.to_parquet(ret_path + f'hfRet_csnorm_{window}d_{mins}min.parquet')

    
def calc_long_short(result):
    sr_l_rtn_no_fee = result['long_short']['long_short_cum_pnl']['long_rtn_no_fee']
    sr_l_rtn_after_fee = result['long_short']['long_short_cum_pnl']['long_rtn_after_fee']
    sr_s_rtn_no_fee = result['long_short']['long_short_cum_pnl']['short_rtn_no_fee']
    sr_s_rtn_after_fee = result['long_short']['long_short_cum_pnl']['short_rtn_after_fee']
    
    sr_s_fee = sr_s_rtn_no_fee - sr_s_rtn_after_fee
    sr_s_rtn_after_fee = sr_s_rtn_no_fee + sr_s_fee
    
    plt.figure(figsize=(20, 5))
    sr_l_rtn_no_fee.plot(label='long_no_fee', color='darkorange')
    sr_l_rtn_after_fee.plot(label='long_after_fee', color='darkorange', linestyle='--')
    sr_s_rtn_no_fee.plot(label='short_no_fee', color='limegreen')
    sr_s_rtn_after_fee.plot(label='short_after_fee', color='limegreen', linestyle='--')
    
    sr_ls_no_fee = sr_l_rtn_no_fee - sr_s_rtn_no_fee
    sr_ls_after_fee = sr_l_rtn_after_fee - sr_s_rtn_after_fee
    
    sr_ls_no_fee.plot(label='long_short_no_fee', color='b')
    sr_ls_after_fee.plot(label='long_short_after_fee', color='b', linestyle='--')
    
    plt.grid()
    plt.legend()
    
    return sr_ls_no_fee, sr_ls_after_fee

def max_drawdown_series(cumulative_returns):
    max_return = cumulative_returns[0]
    max_drawdown_series = []
    for i in range(1, len(cumulative_returns)):
        max_return = max(max_return, cumulative_returns[i])
        drawdown = max_return - cumulative_returns[i]
        max_drawdown_series.append(drawdown)
    return max_drawdown_series
    
if __name__ == '__main__':
    pass