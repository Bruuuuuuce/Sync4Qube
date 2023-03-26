from PqiDataSdk import *
from functools import partial
from itertools import product
from matplotlib import pyplot as plt
from loguru import logger
from tabulate import tabulate
from matplotlib.gridspec import GridSpec
from numba import jit, njit, objmode
import seaborn as sns
import multiprocessing as mp
import pandas as pd
import tqdm
import time
import copy
import numpy as np
import os
import pickle
import getpass
import warnings
import configuration as cfg

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
user = getpass.getuser()


class TradeStatCal():

    def __init__(self, signal_df=None):
        self.ds = PqiDataSdk(user=user, size=1, pool_type="mt", log=False, offline=True)
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        self.signal_df = signal_df
        self.return_type = cfg.return_type.split('_')[0]
        # 获取票池
        self.stock_pool = self.ds.get_ticker_list(date='all')
        for ticker in ['000043', '000022', '601313']:
            self.stock_pool.remove(ticker)
        self.date_list = self.ds.get_trade_dates(start_date=self.start_date, end_date=self.end_date)
        self.index_list = ['hs300', 'zz500', 'zz1000', 'others']
        self.eod_data_dict = self.ds.get_eod_history(tickers = self.stock_pool, start_date = self.start_date, end_date = self.end_date)
        twap_data_dict = self.ds.get_eod_history(tickers=self.stock_pool, start_date=self.start_date, end_date=self.end_date,
                                                 source="ext_stock",
                                                 fields=['TwapBegin30', 'TwapBegin60', 'TwapBegin120', 'Twap'])

        self.eod_data_dict['TwapOpen30'] = twap_data_dict['TwapBegin30'].copy()
        self.eod_data_dict['TwapOpen60'] = twap_data_dict['TwapBegin60'].copy()
        self.eod_data_dict['TwapOpen120'] = twap_data_dict['TwapBegin120'].copy()
        self.eod_data_dict['TwapOpen240'] = twap_data_dict['Twap'].copy()

    def mask_generator(self):
        self.mask_dict = {}
        self.hs300_mask = self.ds.get_index_weight(ticker='000300', start_date=self.start_date, end_date=self.end_date, format='eod').notna().astype('int').replace(0, np.nan)
        self.zz500_mask = self.ds.get_index_weight(ticker='000905', start_date=self.start_date, end_date=self.end_date, format='eod').notna().astype('int').replace(0, np.nan)
        self.zz1000_mask = self.ds.get_index_weight(ticker='000852', start_date=self.start_date, end_date=self.end_date, format='eod').notna().astype('int').replace(0, np.nan)
        self.mask_dict['hs300'] = self.hs300_mask
        self.mask_dict['zz500'] = self.zz500_mask
        self.mask_dict['zz1000'] = self.zz1000_mask

        empty_mask = pd.DataFrame(index=self.stock_pool, columns=self.date_list, dtype='float')
        self.SZ_mask = empty_mask.copy()
        self.SZ_mask.loc[[tik for tik in self.stock_pool if tik.startswith('00') or tik.startswith('30')]] = 1
        self.SH_mask = empty_mask.copy()
        self.SH_mask.loc[[tik for tik in self.stock_pool if tik.startswith('60') or tik.startswith('68')]] = 1
        self.mask_dict['sh'] = self.SH_mask
        self.mask_dict['sz'] = self.SZ_mask

    def cal_index_stat(self):
        self.val_df = self.signal_df / self.signal_df.sum()
        self.vol_df = (self.val_df > 0).astype('int')

        self.val_ratio_df = pd.DataFrame(index=self.date_list, columns=self.index_list, dtype='float')
        self.vol_ratio_df = pd.DataFrame(index=self.date_list, columns=self.index_list, dtype='float')
        self.val_SHSZ_ratio_df = pd.DataFrame(index=self.date_list, columns=['sh', 'sz'], dtype='float')
        self.vol_SHSZ_ratio_df = pd.DataFrame(index=self.date_list, columns=['sh', 'sz'], dtype='float')
        for idx in self.index_list[:-1]:
            self.val_ratio_df[idx] = (self.val_df * self.mask_dict[idx]).sum() / self.val_df.sum()
            self.vol_ratio_df[idx] = (self.vol_df * self.mask_dict[idx]).sum()
        masked_other_val_df = (self.val_df - self.val_df * self.mask_dict['hs300'].fillna(0) - self.val_df * self.mask_dict['zz500'].fillna(0) - self.val_df * self.mask_dict['zz1000'].fillna(0)).replace(0, np.nan)
        masked_other_vol_df = (self.vol_df - self.vol_df * self.mask_dict['hs300'].fillna(0) - self.vol_df * self.mask_dict['zz500'].fillna(0) - self.vol_df * self.mask_dict['zz1000'].fillna(0)).replace(0, np.nan)
        self.val_ratio_df['others'] = masked_other_val_df.sum() / self.val_df.sum()
        self.vol_ratio_df['others'] = masked_other_vol_df.sum()

        self.val_SHSZ_ratio_df['sh'] = (self.val_df * self.mask_dict['sh']).sum() / self.val_df.sum()
        self.val_SHSZ_ratio_df['sz'] = (self.val_df * self.mask_dict['sz']).sum() / self.val_df.sum()
        self.vol_SHSZ_ratio_df['sh'] = (self.vol_df * self.mask_dict['sh']).sum()
        self.vol_SHSZ_ratio_df['sz'] = (self.vol_df * self.mask_dict['sz']).sum()

    def cal_max_tdrat(self):
        self.max_trd_ratio_df = self.val_df.apply(lambda x: -np.sort(-x)).iloc[:3, :].T
        self.max_trd_ratio_df.columns = ['top1', 'top2', 'top3']

    def cal_index_tov(self):
        self.tov_on_index_df = pd.DataFrame(index=self.date_list, columns=self.index_list, dtype='float')
        masked_dict = {}
        for idx in self.index_list[:-1]:
            masked_df = self.val_df * self.mask_dict[idx]
            masked_dict[idx] = masked_df
            turnover = np.abs(masked_df - masked_df.shift(1, axis=1)).sum(axis=0)
            self.tov_on_index_df[idx] = turnover.fillna(0).replace(np.infty, 0)
        masked_other_df = (self.val_df - masked_dict['hs300'].fillna(0) - masked_dict['zz500'].fillna(0) - masked_dict['zz1000'].fillna(0)).replace(0, np.nan)
        turnover = np.abs(masked_other_df - masked_other_df.shift(1, axis=1)).sum(axis=0)
        self.tov_on_index_df['others'] = turnover.fillna(0).replace(np.infty, 0)

    def cal_and_return(self):
        self.mask_generator()
        self.cal_index_stat()
        self.cal_max_tdrat()
        self.cal_index_tov()

        return self.val_ratio_df, self.vol_ratio_df, self.val_SHSZ_ratio_df, self.vol_SHSZ_ratio_df, self.max_trd_ratio_df, self.tov_on_index_df



















