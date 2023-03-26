from PqiDataSdk import *
from functools import partial
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy import stats
from scipy.stats import norm
import config as cfg
import multiprocessing as mp
import xgboost as xgb
import pandas as pd
import tqdm
import time
import copy
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

if __name__ == '__main__':
    ds = PqiDataSdk(user='yhzhou', size=20, pool_type="mp", log=False, offline=True)
    start_date = '20150101'
    end_date = '20211210'
    tickers = ds.get_ticker_list(date='all')
    date_list = ds.get_trade_dates(start_date = start_date, end_date = end_date)
    eod_data_dict = ds.get_eod_history(fields = ['OpenPrice', 'AdjFactor'],
                                       tickers = tickers,
                                       start_date = date_list[0],
                                       end_date = '20211210')

    def read_eod_data(name, path):
        '''读取因子'''
        read_dict = ds.get_eod_feature(fields=[name],
                                       where=path,
                                       tickers=tickers,
                                       dates=date_list)
        return read_dict[name].to_dataframe()

    # 读取因子
    def get_factor_data(factor_list, path, date_list):
        factor_dict = {}
        factor_data_dict = ds.get_eod_feature(fields=factor_list,
                                              where=path,
                                              tickers=tickers,
                                              dates=date_list)
        for fac_name in factor_list:
            if 'raw' in cfg.factor_method:
                factor_dict[fac_name] = factor_data_dict[fac_name].to_dataframe()
            elif 'normal' in cfg.factor_method:
                test_df = factor_data_dict[fac_name].to_dataframe()
                test_df_01 = test_df.rank(pct=True) - 1 / (2 * test_df.rank().max())
                test_df_01.iloc[:, :] = norm.ppf(test_df_01)
                factor_dict[fac_name] = test_df_01.copy()
            elif 'linear' in cfg.factor_method:
                factor_dict[fac_name] = factor_data_dict[fac_name].to_dataframe().rank(pct=True) - 0.5
            else:
                factor_dict[fac_name] = factor_data_dict[fac_name].to_dataframe()

        return factor_dict

    # 读取收益率
    def get_return_data(return_name, return_path):
        return_data = ds.get_eod_feature(fields=[return_name],
                                         where=return_path,
                                         tickers=tickers,
                                         dates=date_list)[return_name].to_dataframe()
        mask = np.where(return_data.isna(), np.nan, 1)
        return return_data, mask

    # 训练集权重
    def get_weights(trainX):
        weight = np.zeros(trainX.shape[0])
        for item in type_list:
            weight = weight + mask_dict[item[0]].stack().loc[trainX.index] * int(item[1])
        return weight

    # 模型训练主体
    def train_n_predict(trainX, trainY, validX, validY, testX, testY, loss, weights=None):
        # 数据封装
        train_xgb = xgb.DMatrix(data=trainX, label=trainY, weight=weights)
        valid_xgb = xgb.DMatrix(data=validX, label=validY)
        test_xgb = xgb.DMatrix(data=testX, label=testY)

        # 模型训练
        XgbPara = cfg.regXgbParams
        evals_result = {}
        watch_list = [(train_xgb, 'train'), (valid_xgb, 'valid')]
        XgbTree = xgb.train(**XgbPara, dtrain=train_xgb, evals=watch_list,  # early_stopping_rounds=50,
                            evals_result=evals_result)  # , obj=eval(loss))

        # 模型预测
        pred_test_y = np.array(XgbTree.predict(test_xgb, ntree_limit=XgbTree.best_ntree_limit))

        pred_test_Y = testY.copy(deep=True)
        pred_test_Y.iloc[:] = pred_test_y
        ic = pd.concat([pred_test_Y, testY], axis=1).corr().values[0, 1]

        pred_test_Y = pred_test_Y.unstack()

        return pred_test_Y, ic


    # 获取X和Y
    def get_Xdata_Y(factor_dict, return_data, date_list, if_train, multi, date=None):
        fac_list = list(factor_dict.keys())
        YData = return_data[date_list].stack(dropna=False)
        if multi:
            res_df = pd.read_csv(f'/home/yhzhou/factor_test_105/multi_period_roll/{date}.csv', index_col=0)
            res_df['score'] = res_df['AlphaSharpeNC'] * 0.005 - 0.5 * res_df['AlphaDrawdownNC'] + res_df['AlphaRetNC'] - res_df['TurnOver'] / 25
            period_result_df = res_df.iloc[:-1, :].sort_values('score', ascending=False)
            period_fac_list = ['eod_' + name for name in list(period_result_df.index)[:250]] + ['eod_shining_fac520']
            XData = pd.DataFrame(index=YData.index, columns=period_fac_list, dtype='float')
            for fac in period_fac_list:
                if fac in fac_list:
                    XData[fac] = factor_dict[fac][date_list].stack(dropna=False)
        else:
            XData = pd.DataFrame(index=YData.index, columns=fac_list, dtype='float')
            for fac in fac_list:
                XData[fac] = factor_dict[fac][date_list].stack(dropna=False)

        if if_train:
            kickout_num = len(np.where(YData.abs() < cfg.kick_out_thrs)[0])
            YData.iloc[:] = np.where(YData.abs() < cfg.kick_out_thrs, np.nan, YData)
            print('kick out ratio: {}%'.format(round(100 * kickout_num / YData.shape[0], 2)))

        # 清nan比例较高的样本
        na_pct = np.sum(np.isnan(XData), axis=1) / XData.shape[1]
        cut_na_points = (na_pct > cfg.reg_na_pct).values.reshape(YData.shape)
        YData.iloc[:] = np.where(cut_na_points, np.nan, YData)

        # 清Y为nan的样本
        na_idx = np.argwhere(~np.isnan(YData.values))
        print(f'Deleted y-NA samples {round((1 - len(na_idx) / YData.shape[0]) * 100, 3)}%')
        XData = XData.iloc[na_idx[:, 0]]
        YData = YData.iloc[na_idx[:, 0]]
        XData = XData.fillna(0)

        return XData, YData

    # 确定日期后的主体
    def model_main_single(train_dates, valid_dates, test_dates, factor_dict, return_data, loss):
        # 确定日期后的主体
        trainX, trainY = get_Xdata_Y(factor_dict, return_data, train_dates, if_train=True, multi=False)
        validX, validY = get_Xdata_Y(factor_dict, return_data, valid_dates, if_train=False, multi=False)
        testX, testY = get_Xdata_Y(factor_dict, return_data, test_dates, if_train=False, multi=False)

        trainY = 100 * trainY
        validY = 100 * validY
        testY = 100 * testY

        weight = get_weights(trainX)
        pred_test_Y, ic = train_n_predict(trainX, trainY, validX, validY, testX, testY, loss=loss)

        return pred_test_Y, ic

    def model_main_multi(train_dates, valid_dates, test_dates, factor_dict, return_data, loss):
        # 确定日期后的主体
        trainX, trainY = get_Xdata_Y(factor_dict, return_data, train_dates, if_train=True, multi=True, date=train_dates[-1])
        validX, validY = get_Xdata_Y(factor_dict, return_data, valid_dates, if_train=False, multi=True, date=train_dates[-1])
        testX, testY = get_Xdata_Y(factor_dict, return_data, test_dates, if_train=False, multi=True, date=train_dates[-1])

        trainY = 100 * trainY
        validY = 100 * validY
        testY = 100 * testY

        pred_test_Y, ic = train_n_predict(trainX, trainY, validX, validY, testX, testY, loss=loss)

        return pred_test_Y, ic


    def main_loop(train_start, valid_periods, test_periods, loss, start_date, end_date, factor_list, path, return_name,
                  return_path):
        # 主循环
        total_start = time.time()
        date_list = ds.get_trade_dates(start_date=start_date, end_date=end_date)

        # 读数据
        factor_dict = get_factor_data(factor_list=factor_list, path=path, date_list=date_list)
        return_data, mask = get_return_data(return_name=return_name, return_path=return_path)
        predict_return_data = pd.DataFrame(index=return_data.index, columns=return_data.columns, dtype='float')

        # 训模型&预测
        model_date_list = ds.get_trade_dates(start_date=start_date, end_date=end_date)
        valid_start = train_start
        test_start = train_start + max(cfg.return_len, valid_periods) + 3
        test_end = test_start + test_periods
        while test_start < len(model_date_list):
            t_start = time.time()
            if 'expand' in cfg.reg_mode:
                train_dates = model_date_list[:valid_start]
            elif 'roll' in cfg.reg_mode:
                train_dates = model_date_list[valid_start - train_start:valid_start]
            else:
                print('Suggest Re-Run the experiment as wrong type for mode.')
                train_dates = model_date_list[:valid_start]
            valid_dates = model_date_list[valid_start:test_start - 3]
            test_dates = model_date_list[test_start:test_end]
            print('Start training {} to {}.'.format(train_dates[0], train_dates[-1]))

            if 'single' in cfg.reg_switch:
                pred_test_Y, ic = model_main_single(train_dates, valid_dates, test_dates, factor_dict, return_data, loss)
            elif 'multi' in cfg.reg_switch:
                pred_test_Y, ic = model_main_multi(train_dates, valid_dates, test_dates, factor_dict, return_data, loss)
            else:
                pred_test_Y, ic = model_main_single(train_dates, valid_dates, test_dates, factor_dict, return_data, loss)

            predict_return_data[test_dates] = pred_test_Y

            print('{} to {} has been predicted. Time used: {}s. IC: {}'.format(test_dates[0], test_dates[-1],
                                                                               round(time.time() - t_start, 2),
                                                                               round(ic, 4)))
            valid_start += test_periods
            test_start += test_periods
            test_end += test_periods

        predict_return_data = predict_return_data * mask
        print('Total IC: {}'.format(round(predict_return_data.corrwith(return_data).mean(), 4)))

        # 存因子
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M%S', time.localtime(time.time()))
        diy_name = f'eod_xgb_{curr_date + curr_time}_reg'
        ds.save_eod_feature(data={diy_name: predict_return_data},
                            where='/home/shared/Data/data/shared/low_fre_alpha/yhzhou_comb_factors',
                            feature_type='eod',
                            encrypt=False)
        print('name: {}, time used: {}s'.format(diy_name, round(time.time() - total_start, 2)))

        return predict_return_data, diy_name

    hs300_df = read_eod_data('eod_hs300', '/data/shared/low_fre_alpha/index_data').notna().astype('int')
    zz500_df = read_eod_data('eod_zz500', '/data/shared/low_fre_alpha/index_data').notna().astype('int')
    zz1000_df = read_eod_data('eod_zz1000', '/data/shared/low_fre_alpha/index_data').notna().astype('int')
    mask_dict = {}
    mask_dict['TA'] = hs300_df.copy()
    mask_dict['TB'] = zz500_df.copy()
    mask_dict['TC'] = zz1000_df.copy()
    mask_dict['TD'] = pd.DataFrame(1, index=tickers, columns=date_list, dtype='float')

    tasks = ['TD*1+TA*1', 'TD*1+TA*2', 'TD*1+TA*3', 'TD*1+TA*5',
             'TD*1+TB*1', 'TD*1+TB*2', 'TD*1+TB*3', 'TD*1+TB*5',
             'TD*1+TC*1', 'TD*1+TC*2', 'TD*1+TC*3', 'TD*1+TC*5',
             'TA*2+TB*1+TC*1', 'TA*3+TB*1+TC*1', 'TA*4+TB*1+TC*1', 'TA*6+TB*1+TC*1',
             'TA*1+TB*2+TC*1', 'TA*1+TB*3+TC*1', 'TA*1+TB*4+TC*1', 'TA*1+TB*6+TC*1',
             'TA*1+TB*1+TC*2', 'TA*1+TB*1+TC*3', 'TA*1+TB*1+TC*4', 'TA*1+TB*1+TC*6']

    result_dict = {}
    for type in tasks:
        type_list = [i.split('*') for i in type.split('+')]
        print(type_list)
        predict_return_data, diy_name = main_loop(train_start=240,
                                                  valid_periods=10,
                                                  test_periods=40,
                                                  loss='xgbreg_loss_function_mse',
                                                  start_date='20160101',
                                                  end_date='20210301',
                                                  factor_list=cfg.reg_factor_list,
                                                  path=cfg.reg_factor_path,
                                                  return_name=cfg.return_name,
                                                  return_path=cfg.return_path)
        result_dict[type] = diy_name
        print('{} name: {}'.format(type, diy_name))
        print('============================================================================================\n\n')
    print(pd.DataFrame([result_dict]))

























