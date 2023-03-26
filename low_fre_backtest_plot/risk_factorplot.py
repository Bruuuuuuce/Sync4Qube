"""
绘制风险归因的各项函数（从原RiskPlot project抽离而来）
TODO: 清除与AlphaTest重叠的部分
"""

# load packages
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import time
import configuration as cfg
from PqiDataSdk import *

import sys
import getpass
USER = getpass.getuser()

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11



# TODO: 从外部读取这些东东，不要在函数里面读取（方便batch的shm）
class SimplifiedRiskAnal:
    def __init__(self, benchmark_index='000852', signal_df=None):
        self.bm_index = benchmark_index
        self.myconnector = PqiDataSdk(user=USER, size=20, pool_type="mp", log=False, offline=True, str_map=False)
        self.signal_df = signal_df
        self.anal_method = cfg.signal_type
        self.start_date = self.signal_df.columns[0]
        self.end_date = self.signal_df.columns[-1]
        self.date_list = self.myconnector.get_trade_dates(start_date=self.start_date, end_date=self.end_date)
        self.ind_name_matcher = {
            'ind_1': 'Transportation', 'ind_2': 'LeisureService', 'ind_3': 'Media', 'ind_4': 'Utility',
            'ind_5': 'Agriculture', 'ind_6': 'Chem', 'ind_7': 'Bio-Medical',
            'ind_8': 'Trading', 'ind_9': 'Defence', 'ind_10': 'HomeApp', 'ind_11': 'Constrct-Material',
            'ind_12': 'Constrct-Decorate', 'ind_13': 'RealEst', 'ind_14': 'Metals', 'ind_15': 'Machinery',
            'ind_16': 'Auto', 'ind_17': 'E-Components', 'ind_18': 'E-Equip', 'ind_19': 'Textile', 'ind_20': 'Mixed',
            'ind_21': 'Computer', 'ind_22': 'LightMnfac', 'ind_23': 'Commun', 'ind_24': 'Mining',
            'ind_25': 'Steel', 'ind_26': 'Banks', 'ind_27': 'OtherFinance', 'ind_28': 'FoodBvrg'
        }
        self.idx_list_all = ['000016', '000300', '000905', '000852']
        self.stock_pool = self.myconnector.get_ticker_list(date='all')
        for ticker in ['000043', '000022', '601313']:
            self.stock_pool.remove(ticker)
        self.next_end_date = self.myconnector.get_next_trade_date(trade_date=self.end_date)
        self.eod_data_dict = self.myconnector.get_eod_history(tickers=self.stock_pool, start_date=self.start_date,
                                                              end_date=self.end_date)
        self.idx_data_dict = self.myconnector.get_eod_history(tickers=self.idx_list_all,
                                                              start_date=self.start_date,
                                                              end_date=self.end_date, source='index')
        self.signal_df = (self.eod_data_dict['ClosePrice'] * 0 + self.signal_df).fillna(0)
        print('Factor Analysis Initialized.')
        # 大类风格因子值
        self.risk_fac_list = ['beta', 'book_to_price', 'earnings_yield', 'growth', 'leverage', 'liquidity', 'momentum', 'non_linear_size', 'residual_volatility', 'size']
        self.risk_expo_dict = self.myconnector.get_factor_exposure(start_date=self.start_date,
                                                                   end_date=self.next_end_date,
                                                                   factors=self.risk_fac_list)
        for fac_name in self.risk_expo_dict.keys():
            self.risk_expo_dict[fac_name] = self.risk_expo_dict[fac_name].shift(-1, axis=1)[self.date_list]

        # 因子收益率矩阵
        self.risk_ret_df = self.myconnector.get_factor_return(start_date=self.start_date,
                                                              end_date=self.next_end_date,
                                                              factors=self.risk_fac_list).shift(-1, axis=1)[self.date_list]
        # 行业分类矩阵
        self.get_ind_related()

        # 获得指数权重
        self.idx_weight_dict = {}
        for idx in self.idx_list_all:
            self.idx_weight_dict[idx] = self.myconnector.get_index_weight(ticker=idx,
                                                                          start_date=self.start_date,
                                                                          end_date=self.next_end_date,
                                                                          format='eod').shift(-1, axis=1)[self.date_list]

        # 处理权重
        weight = self.signal_df / self.signal_df.sum()
        prc_df = self.eod_data_dict['ClosePrice'].copy()
        if self.anal_method == 'vol':
            self.w = (weight * prc_df).dropna(axis=0)[self.date_list]
        else:
            self.w = weight[self.date_list]
        self.w = self.w / self.w.sum()
        self.w = self.w.fillna(0)

        # 读bm权重
        self.w_bm = self.idx_weight_dict[self.bm_index]
        self.w_bm = self.w_bm.loc[self.w.index, :]
        self.w_bm = self.w_bm / self.w_bm.sum()
        self.w_bm = self.w_bm.fillna(0)


    def get_ind_related(self):
        # 获取行业分类矩阵
        self.ind_name_en_lst = ['transportation', 'media', 'utility', 'agriculture', 'biomedicine', 'retail',
                                'military', 'basic_chemicals', 'household_appliance', 'arch_mat', 'arch_deco',
                                'real_estate', 'nonferrous_metals', 'machinery_n_eqpt',
                                'automobile', 'coal', 'env_prot', 'electrical_eqpt', 'electronics', 'petrochem',
                                'social_services', 'textiles_n_apparel', 'comprehensive', 'beauty', 'computer',
                                'light_mfg', 'telecom', 'steel', 'bank', 'non_bank_finance', 'food_n_beverage']
        self.ind_expo_dict = self.myconnector.get_factor_exposure(tickers=self.stock_pool, start_date=self.start_date,
                                                           end_date=self.next_end_date,
                                                           factors=self.ind_name_en_lst)

        for ind_name in self.ind_expo_dict.keys():
            self.ind_expo_dict[ind_name] = self.ind_expo_dict[ind_name].shift(-1, axis=1)

        # 米匡接入行业收益率数据
        self.ind_ret_df = self.myconnector.get_factor_return(start_date=self.start_date,
                                                             end_date=self.next_end_date,
                                                             factors=self.ind_name_en_lst).shift(-1, axis=1)[self.date_list]

    def calc_attribution(self):
        # 计算风格因子归因
        self.style_attribute = pd.DataFrame(index=self.risk_fac_list, columns=self.date_list, dtype='float')
        self.style_exposure = pd.DataFrame(index=self.risk_fac_list, columns=self.date_list, dtype='float')
        self.ind_attribute = pd.DataFrame(index=self.ind_name_en_lst, columns=self.date_list, dtype='float')
        self.ind_exposure = pd.DataFrame(index=self.ind_name_en_lst, columns=self.date_list, dtype='float')
        for fac_name in self.risk_expo_dict.keys():
            self.style_exposure.loc[fac_name] = (self.risk_expo_dict[fac_name] * (self.w - self.w_bm)).sum()
            self.style_attribute.loc[fac_name] = self.style_exposure.loc[fac_name] * self.risk_ret_df.loc[fac_name]

        for ind_name in self.ind_name_en_lst:
            self.ind_exposure.loc[ind_name] = (self.ind_expo_dict[ind_name] * (self.w - self.w_bm)).sum()
            self.ind_attribute.loc[ind_name] = self.ind_exposure.loc[ind_name] * self.ind_ret_df.loc[ind_name]

class risk_plotter:
    def __init__(self, benchmark_index='000852'):
        self.benchmark_index = benchmark_index
        # TODO: 对标票池的修改，可以从dataassist直接传进来

    def output_riskdf(self, csv_name='test', signal_type='val', long_signal_df=None, res_path='./df_res/', is_long=True):
        """
        riskplot绘图
        :param fig_name: 存储的因子名
        :param csv_paths: 一个list，装有读取持仓信号的路径或者dataframe
        :param saving_path: 存储最终因子图的路径
        :param is_long: True if it is long holdings, False otherwise. （只和最后导出的图的命名有关）
        """

        # 导入数据 & 处理数据
        # 将调仓周期内的所有路径的信号相加取平均
        calculator = SimplifiedRiskAnal(
                      benchmark_index=self.benchmark_index,
                      signal_df=long_signal_df
                    )
        calculator.calc_attribution()

        # 存行业暴露、收益
        calculator.ind_exposure.index = calculator.ind_name_en_lst
        calculator.ind_attribute.index = calculator.ind_name_en_lst

        # 返回四个df
        style_rtn_ts_df = calculator.style_attribute.cumsum(axis=1).T
        style_exposure_ts_df = calculator.style_exposure.T

        industry_exposure_df = pd.concat([calculator.ind_exposure.mean(axis=1), calculator.ind_attribute.sum(axis=1)], axis=1)
        industry_exposure_df.columns = ['exposure', 'attribute']
        industry_exposure_df = industry_exposure_df.sort_values(cfg.rank_method, ascending=False)

        style_exposure_df = pd.concat([calculator.style_exposure.mean(axis=1), calculator.style_attribute.sum(axis=1)], axis=1)
        style_exposure_df.columns = ['exposure', 'attribute']
        style_exposure_df = style_exposure_df.loc[['size', 'non_linear_size', 'beta', 'liquidity', 'momentum', 'growth', 'leverage', 'book_to_price', 'earnings_yield', 'residual_volatility']]

        return style_rtn_ts_df, style_exposure_ts_df, industry_exposure_df, style_exposure_df









