import numpy as np
from PqiDataSdk import *
from functools import partial
from itertools import product
from matplotlib import pyplot as plt
from loguru import logger
from tabulate import tabulate
from matplotlib.gridspec import GridSpec
from numba import jit, njit, objmode
from combo_backtest import ComboBT
from combo_plot import Template
import seaborn as sns
import multiprocessing as mp
import pandas as pd
import tqdm
import time
import copy
import os
import pickle
import datetime
import getpass
import warnings

import configuration as cfg
from risk_factorplot import risk_plotter
from trading_stat_cal import TradeStatCal
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
user = getpass.getuser()

if __name__ == '__main__':

    if 'combo' in cfg.test_type:
        total_name = '{}_bm_{}_{}_{}'.format(cfg.combo_name, cfg.index, cfg.return_type, cfg.head)
    else:
        total_name = '{}_bm_{}_{}'.format(cfg.signal_name, cfg.index, cfg.return_type)
    print('{} test starts.'.format(total_name))

    # 读取因子
    total_start = time.time()
    ds = PqiDataSdk(user=user, size=1, pool_type="mt", log=False, offline=True)
    date_list = ds.get_trade_dates(start_date=cfg.start_date, end_date=cfg.end_date)
    tickers = ds.get_ticker_list(date='all')
    for ticker in ['000043', '000022', '601313']:
        tickers.remove(ticker)

    ''' 
    ================================ 组合回测 ================================
    '''
    start_time = time.time()
    ComboBT = ComboBT()
    ComboBT.data_prepare()
    if 'combo' in cfg.test_type:
        print('开始回测组合.')
        combo_df = ds.get_eod_feature(fields=[cfg.combo_name],
                                      where=cfg.combo_path,
                                      tickers=tickers,
                                      dates=ds.get_trade_dates(start_date=cfg.total_start_date, end_date=cfg.total_end_date))[cfg.combo_name].to_dataframe()
        data_dict, combo_long_result, combo_long_signal_df, drawdown_tuple = ComboBT.backtest(factor_df = combo_df,
                                                                                              name = cfg.combo_name,
                                                                                              head = cfg.head,
                                                                                              method = cfg.method,
                                                                                              cost = cfg.cost,
                                                                                              group_num = cfg.group_num,
                                                                                              benchmark = cfg.benchmark,
                                                                                              index = cfg.index,
                                                                                              return_type = cfg.return_type,
                                                                                              start_date = cfg.start_date,
                                                                                              end_date = cfg.end_date,
                                                                                              plot = True)

    else:
        print('开始回测信号.')
        signal_df = ds.get_eod_feature(fields=[cfg.signal_name],
                                       where=cfg.signal_path,
                                       tickers=tickers,
                                       dates=ds.get_trade_dates(start_date=cfg.total_start_date, end_date=cfg.total_end_date))[cfg.signal_name].to_dataframe()
        data_dict, combo_long_result, combo_long_signal_df, drawdown_tuple = ComboBT.signal_backtest(signal_df = signal_df,
                                                                                                     name = cfg.signal_name,
                                                                                                     cost = cfg.cost,
                                                                                                     benchmark = cfg.benchmark,
                                                                                                     index = cfg.index,
                                                                                                     return_type = cfg.return_type,
                                                                                                     start_date = cfg.start_date,
                                                                                                     end_date = cfg.end_date,
                                                                                                     plot = True)

    excess_500_tuple = (combo_long_result[0] - combo_long_result[2], combo_long_result[1] - combo_long_result[2])
    annual_result_df = ComboBT.annual_stat(excess_500_tuple, int(cfg.start_date[:4]), int(cfg.end_date[:4]))
    rtn_df = pd.concat([combo_long_result[2], combo_long_result[3], combo_long_result[1] - combo_long_result[2], combo_long_result[1] - combo_long_result[3]], axis=1).cumsum()
    rtn_df.columns = ['index_{}'.format(cfg.index), 'index_pool', 'alpha_{}'.format(cfg.index), 'alpha_pool']
    print('信号回测部分完毕. 耗时: {}s.'.format(round(time.time() - start_time, 3)))

    '''
    ================================ 风格分析 ================================
    '''

    print('开始风格分析.')
    start_time = time.time()
    rp = risk_plotter(benchmark_index = cfg.bm_index)

    # 读取信号
    sig_df = combo_long_signal_df.copy()
    sig_df = sig_df.dropna(how='all', axis=1)
    sig_df.iloc[:, :] = np.where(sig_df > 0, sig_df, 0)

    # 判断测试类型
    if 'single' in cfg.signal_test_type:
        sig_df = sig_df[[cfg.single_date]]
    elif 'multi' in cfg.signal_test_type:
        sig_df = sig_df.loc[:, cfg.multi_start_date:cfg.multi_end_date]
    else:
        print('Wrong Type for signal test. Only single or multi is permitted.')
        sig_df = None

    # 输出表格
    style_rtn_ts_df, style_exposure_ts_df, industry_exposure_df, style_exposure_df = rp.output_riskdf(csv_name=total_name,
                                                                                                      signal_type='val',
                                                                                                      long_signal_df=sig_df,
                                                                                                      is_long=True)
    print('风格分析部分完毕. 耗时: {}s.'.format(round(time.time() - start_time, 3)))

    '''
    ================================ 生成所有需要的矩阵 ================================
    '''

    print('开始生成矩阵.')
    start_time = time.time()
    TradingCal = TradeStatCal(signal_df=combo_long_signal_df.shift(1, axis=1))
    stat_res = TradingCal.cal_and_return()

    val_ratio_df = stat_res[0]
    vol_ratio_df = stat_res[1]
    val_SHSZ_ratio_df = stat_res[2]
    vol_SHSZ_ratio_df = stat_res[3]
    max_trd_ratio_df = stat_res[4]
    tov_on_index_df = stat_res[5]
    print('矩阵生成部分完毕. 耗时: {}s.'.format(round(time.time() - start_time, 3)))

    '''
    ================================ 存数据pickle ================================
    '''
    today = datetime.datetime.now().strftime(format='%Y%m%d')
    save_path = cfg.save_fig_path + 'res_{}/'.format(today)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    pickle_for_save = {'stat_by_year': {'df': annual_result_df, 'title': 'Annual Stat'},
                        'style_rtn_ts_dict': {'df': style_rtn_ts_df, 'title': 'Risk Factor Attribute c2nextc'},
                        'style_exposure_ts_dict': {'df': style_exposure_ts_df, 'title': 'Risk Factor Exposure c2nextc'},
                        'industry_exposure_dict': {'df': industry_exposure_df, 'title': 'Industry Exposure and Return Attribution c2nextc'},
                        'style_exposure_dict': {'df': style_exposure_df, 'title': 'Risk Factor Exposure and Return Attribution c2nextc'},
                        'max_trd_ratio_dict': {'df': max_trd_ratio_df, 'title': 'Holding Weight in Indexes c2nextc'},
                        'tov_on_index_dict': {'df': tov_on_index_df, 'title': 'TOV on Index'},
                        'val_SHSZ_ratio_dict': {'df': val_SHSZ_ratio_df, 'title': 'val SHSZ ratio'},
                        'vol_SHSZ_ratio_dict': {'df': vol_SHSZ_ratio_df, 'title': 'stock count SHSZ'},
                        'val_ratio_dict': {'df': val_ratio_df, 'title': 'val index ratio'},
                        'vol_ratio_dict': {'df': vol_ratio_df, 'title': 'stock count index'},
                        'drawdown_index_dict': {'df': drawdown_tuple[0], 'title': 'max drawdown index'},
                        'drawdown_pool_dict': {'df': drawdown_tuple[1], 'title': 'max drawdown pool'},
                        'rtn_dict': {'df': rtn_df, 'title': 'Return Ts'},
                        'save_path': save_path + '{}.jpg'.format(total_name)
                       }
    with open('./save_data/{}.pkl'.format(total_name), 'wb') as f:
        pickle.dump(pickle_for_save, f)

    '''
    ================================ 画回测大图 ================================
    '''

    print('开始画总图.')
    start_time = time.time()
    # 实例化模板类，设置长宽以及日期采样间隔
    template = Template(
        fig_width=50,
        fig_height=12,
        sample_interval=int(len(date_list) / 8)
    )

    # 默认日间统计图模板
    template.cross_day_plot(
        stat_by_year=pickle_for_save['stat_by_year'],
        style_rtn_ts_dict=pickle_for_save['style_rtn_ts_dict'],
        style_exposure_ts_dict=pickle_for_save['style_exposure_ts_dict'],
        industry_exposure_dict=pickle_for_save['industry_exposure_dict'],
        style_exposure_dict=pickle_for_save['style_exposure_dict'],
        max_trd_ratio_dict=pickle_for_save['max_trd_ratio_dict'],
        tov_on_index_dict=pickle_for_save['tov_on_index_dict'],
        val_SHSZ_ratio_dict=pickle_for_save['val_SHSZ_ratio_dict'],
        vol_SHSZ_ratio_dict=pickle_for_save['vol_SHSZ_ratio_dict'],
        val_ratio_dict=pickle_for_save['val_ratio_dict'],
        vol_ratio_dict=pickle_for_save['vol_ratio_dict'],
        rtn_dict=pickle_for_save['rtn_dict'],
        drawdown_index_dict=pickle_for_save['drawdown_index_dict'],
        drawdown_pool_dict=pickle_for_save['drawdown_pool_dict'],
        save_path=pickle_for_save['save_path']
    )
    print('画图完毕. 耗时: {}s.'.format(round(time.time() - start_time, 3)))

    print('所有流程结束. 耗时: {}s.'.format(round(time.time() - total_start, 3)))

















