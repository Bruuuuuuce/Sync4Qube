# 因子函数文件
from unittest import result
import numpy as np
import pandas as pd
import tqdm
import pickle
import os
import re
import time
from itertools import product
import warnings

warnings.filterwarnings('ignore')

RANKIC = False

tt = time.time


def np_rank(arr: np.array, axis: int, ascending: bool = True) -> np.array:
    """返回排名

    Args:
        arr (np.array): 原序列
        axis (int): 排名轴
        ascending (bool, optional): 是否升序. Defaults to True.

    Returns:
        np.array: 排名
    """
    arr = arr if ascending else -arr
    res = np.argsort(np.argsort(arr, axis=axis), axis=axis).astype(float)
    res[np.isnan(arr)] = np.nan
    return res + 1


def np_corrwith(arr1: np.array, arr2: np.array) -> np.array:
    """返回相关系数

    Args:
        arr1 (np.array): 1*m*n or m*n
        arr2 (np.array): k*m*n

    Returns:
        np.array: k*n
    """
    # tile
    arr2 = arr2.copy()
    arr1 = np.tile(arr1, reps=(arr2.shape[0], 1, 1))
    arr1[np.isnan(arr2)], arr2[np.isnan(arr1)] = np.nan, np.nan
    # decentral
    arr1 = arr1 - np.nanmean(arr1, axis=1, keepdims=True)
    arr2 = arr2 - np.nanmean(arr2, axis=1, keepdims=True)
    dot = np.nansum(arr1 * arr2, axis=1)
    var1 = np.nansum(np.square(arr1), axis=1)
    var2 = np.nansum(np.square(arr2), axis=1)
    return dot / np.sqrt(var1 * var2)


def mins_backtest(inputs, params, context):
    '''
    输入: 因子, rtn和mask; 参数params: 分组数
    输出: 每天一个pkl, 包含一个字典, 键是因子名, 取出后result[fac]也是字典, result[fac]["mins_result"]是该因子每分钟数值的df, 
          行是字段, 列是时间; result[fac]["common_result"]是不同票池和rtn的常规统计, 格式为字典, 键是mask, 其中30min/60min的
          rtn分日内和隔夜统计一日均值
    '''

    mins_data, rtn_data, mask_data = inputs
    mins_arr, mins_map = mins_data
    rtn_arr, rtn_map = rtn_data
    mask_arr, mask_map = mask_data

    features_list = [
        item for item in mins_map.keys() if item not in ['TimeStamp', 'ticker']
    ]
    rtn_list = [
        item for item in rtn_map.keys() if item not in ['TimeStamp', 'ticker']
    ]
    mask_list = [
        item for item in mask_map.keys() if item not in ['TimeStamp', 'ticker']
    ]  # mask列表, 使用因子全名

    stock_mins_time = mins_arr[0][mins_map['TimeStamp']][0]  # 读取的TimeStamp
    stock_mins_ticker = mins_arr[0][mins_map['ticker']][:, 0]  # 读取的tickers

    fee_rate = 0.0015
    group_num = params["group_num"]  # 分组数

    # rtn数据标识
    rtn_short_name_dict = {}  # 因子短名到如何统计日内和隔夜的映射
    rtn_short_name_list = [rtn_name.split(".")[-1] for rtn_name in rtn_list]
    rtn_short_name_list_ext = []
    for rtn_short_name in rtn_short_name_list:
        rtn_len_list = re.findall(r'\d+', rtn_short_name)
        if len(rtn_len_list) < 3:
            rtn_short_name_dict[rtn_short_name] = []
            rtn_short_name_list_ext.append(rtn_short_name)
        else:
            intra_end = 240 - (int(rtn_len_list[-2]) + int(rtn_len_list[-1])
                              )  # 日内统计结束时间
            ovn_start = 240 - int(rtn_len_list[-2])  # 隔夜统计开始时间
            rtn_short_name_dict[rtn_short_name] = [intra_end, ovn_start]
            rtn_short_name_list_ext += [
                f'{rtn_short_name}_intra', f'{rtn_short_name}_ovn'
            ]

    # 整合所有rtn数据得到高维矩阵，dim1表示rtn种类，dim2表示ticker，dim3表示mins
    lst = [rtn_map[name] for name in rtn_list]
    num_rtn = len(lst)
    all_rtn_arr: np.array = rtn_arr[0][lst].astype(float)
    all_rtn_arr[np.isinf(all_rtn_arr)] = np.nan

    # 整合所有features数据得到高维矩阵，dim1表示features种类，dim2表示ticker，dim3表示mins
    lst = [mins_map[name] for name in features_list]
    num_feature = len(lst)
    all_feature_arr: np.array = mins_arr[0][lst].astype(float)
    all_feature_arr[np.isinf(all_feature_arr)] = np.nan

    # 整合所有mask数据
    lst = [mask_map[name] for name in mask_list]
    mask_arr: np.array = mask_arr[0][lst].astype(float)
    mask_arr[mask_arr == 0] = np.nan

    result_dict = {}  # 存储结果
    """分市场计算回报，统计日间"""
    mask_res_dict = {}
    mins_res_dict = {}
    for i in range(mask_arr.shape[0] + 1):
        # for i in [mask_arr.shape[0]]:
        if i == mask_arr.shape[0]:  # 全市场
            mask, mask_name = 1, 'all'
        else:
            mask, mask_name = mask_arr[i], mask_list[i]
        # 筛选当前市场下的ticker
        mask_feature_arr = all_feature_arr * mask
        mask_rtn_arr = all_rtn_arr * mask
        # tmp = np.isnan(mask_feature_arr) | np.isnan(mask_rtn_arr)
        # mask_feature_arr[tmp], mask_rtn_arr[tmp] = np.nan, np.nan
        if RANKIC:
            mask_feature_rank = np_rank(mask_feature_arr, axis=1)  # 截面排序
            mask_rtn_rank = np_rank(mask_rtn_arr, axis=1)  # 截面排序
        # 计算多空头仓位权重
        feature_demean = mask_feature_arr - np.nanmean(
            mask_feature_arr, axis=1, keepdims=True)  # 特征去中心
        feature_demean = feature_demean / np.nansum(
            np.abs(feature_demean), axis=1, keepdims=True) * 2  # 线性变换不影响IC
        l, s = feature_demean > 0, feature_demean < 0
        # 为快速计算corr
        if RANKIC:
            feature_rank_demean = mask_feature_rank - np.nanmean(
                mask_feature_rank, axis=1, keepdims=True)  # nf * tickers * mins
            feature_rank_demean_square_sum_sqrt = np.sqrt(
                np.nansum(np.square(feature_rank_demean), axis=1))
            rtn_rank_demean = mask_rtn_rank - np.nanmean(
                mask_rtn_rank, axis=1, keepdims=True)  # nr * tickers * mins
            rtn_rank_demean_square_sum_sqrt = np.sqrt(
                np.nansum(np.square(rtn_rank_demean), axis=1))  # nr * mins
        # feature_demean_square_sum_sqrt = np.sqrt(np.nansum(np.square(feature_demean), axis=1)) # nf * mins
        rtn_demean = mask_rtn_arr - np.nanmean(
            mask_rtn_arr, axis=1, keepdims=True)  # nr * tk * m
        # rtn_demean_square_sum_sqrt = np.sqrt(np.nansum(np.square(rtn_demean), axis=1)) # nr*mins
        # tile
        # tile_rtn_demean = np.tile(rtn_demean,reps=(num_feature,1,1,1)) # nf*nr*tk*m
        # tile_demean_dot = tile_rtn_demean * feature_demean[:,None,:,:] # nf*nr*tk*m
        # tile_demean_dot_sum = np.nansum(tile_demean_dot, axis=2) # nf*nr*m
        # tile_long_rtn = np.nansum(tile_demean_dot * (feature_demean[:,None,:,:]>0), axis=2)
        # tile_short_rtn =  tile_long_rtn - tile_demean_dot_sum
        # tile_std_dot = np.tile(rtn_demean_square_sum_sqrt, reps=(num_feature,1,1)) * feature_demean_square_sum_sqrt[:,None,:]
        # tile_ic = tile_demean_dot_sum / tile_std_dot # nf * nr * m
        # 遍历所有因子
        res_lst = []
        for rtn_index in range(num_rtn):
            rtn_short_name = rtn_short_name_list[rtn_index]
            # 多空头收益率 与 IC
            long_rtn_srs = np.nansum(rtn_demean[rtn_index][None, :, :] *
                                     feature_demean * l,
                                     axis=1)
            short_rtn_srs = -np.nansum(
                rtn_demean[rtn_index][None, :, :] * feature_demean * s, axis=1)
            # long_rtn_srs = tile_long_rtn[:,rtn_index,:]
            # short_rtn_srs = tile_short_rtn[:,rtn_index,:]
            ic_srs = np_corrwith(mask_rtn_arr[rtn_index], mask_feature_arr)
            if RANKIC:
                rankic_srs = np.nansum(rtn_rank_demean[rtn_index][None,:,:] * feature_rank_demean, axis = 1) / \
                    (rtn_rank_demean_square_sum_sqrt[rtn_index][None,:] * feature_rank_demean_square_sum_sqrt)
            else:
                rankic_srs = np.zeros(ic_srs.shape)
            if rtn_short_name == 'm1_ts_stock_y_fix60min_5minTwap_rtn' and mask_name == 'all':
                mins_res_dict['IC'] = ic_srs
                mins_res_dict['rankIC'] = rankic_srs
                mins_res_dict['LongRtn'] = long_rtn_srs
                mins_res_dict['ShortRtn'] = short_rtn_srs
                mins_res_dict['feature_demean'] = feature_demean
                mins_res_dict['rtn_demean'], mins_res_dict['rtn'] = rtn_demean[
                    rtn_index][None, :, :], all_rtn_arr[rtn_index][None, :, :]
            # 判断是否需要区分日内与隔夜
            func = lambda arr_lst: np.vstack(
                [np.nanmean(arr, axis=1) for arr in arr_lst])
            if len(rtn_short_name_dict[rtn_short_name]) == 0:
                res_lst.append(
                    func([ic_srs, rankic_srs, long_rtn_srs, short_rtn_srs]))
            else:
                intra_end, ovn_start = rtn_short_name_dict[rtn_short_name]
                intra = func([
                    ic_srs[:, :intra_end], rankic_srs[:, :intra_end],
                    long_rtn_srs[:, :intra_end], short_rtn_srs[:, :intra_end]
                ])
                ovn = func([
                    ic_srs[:, ovn_start:], rankic_srs[:, ovn_start:],
                    long_rtn_srs[:, ovn_start:], short_rtn_srs[:, ovn_start:]
                ])
                res_lst += [intra, ovn]
        mask_res_dict[mask_name] = np.array(
            res_lst)  # dim1 rtn; dim2 IC,rankIC,...; dim3 feature
    """精细统计, 5min-60min Twap"""
    ovn_rtn = all_rtn_arr[rtn_short_name_list.index("m1_ts_stock_y_ovn_rtn")]
    mins_res_dict['ovn_IC'] = np_corrwith(ovn_rtn, all_feature_arr)
    # 费前分组超额收益
    rtn_arr = mins_res_dict['rtn']
    rtn_demean = mins_res_dict['rtn_demean']
    feature_qt = np.nanquantile(all_feature_arr,
                                np.linspace(0, 1, group_num + 1)[::-1],
                                axis=1)
    for i in range(group_num):
        upper, lower = feature_qt[i][:, None, :], feature_qt[i + 1][:, None, :]
        feature_selected = ((lower < all_feature_arr) &
                            (all_feature_arr <= upper)).astype(float)
        mins_res_dict[f"Group{i}"] = np.nansum(
            rtn_demean * feature_selected, axis=1) / np.nansum(feature_selected,
                                                               axis=1)
    # feature_descend_rank = np_rank(all_feature_arr, axis=1, ascending=False)
    # num_per_group = np.nanmax(feature_descend_rank, axis=1,keepdims=True) / group_num
    # print(num_per_group)
    # for i in range(group_num):
    #     signal = ((feature_descend_rank <= (i+1) * num_per_group) & (feature_descend_rank > i * num_per_group)).astype(float)
    #     signal[signal == 0] = np.nan
    #     # print(signal)
    #     ret = np.nansum(signal * rtn_demean, axis=1) / np.nansum(signal, axis = 1)
    #     mins_res_dict[f"Group{i}"] = ret
    # 准确度
    feature_demean = mins_res_dict['feature_demean']
    l, s = feature_demean > 0, feature_demean <= 0
    mins_res_dict["LongPrecision"] = np.nansum(
        ((l) & (rtn_demean > 0)), axis=1) / np.nansum(l, axis=1)
    mins_res_dict["ShortPrecision"] = np.nansum(
        ((s) & (rtn_demean < 0)), axis=1) / np.nansum(s, axis=1)
    """整合结果，使得符合原格式"""
    # mask_res_dict
    result_index = [
        "IC", "rankIC", "ovn_IC", "LongRtn", "ShortRtn", "LongPrecision",
        "ShortPrecision"
    ]
    result_dict = dict([[fn, {"common_result": {}}] for fn in features_list])
    for mask_name, res in mask_res_dict.items():
        for i, feature_name in enumerate(features_list):
            result_dict[feature_name]["common_result"][
                mask_name] = pd.DataFrame(
                    res[:, :, i].T,
                    columns=rtn_short_name_list_ext,
                    index=["IC", "rankIC", "LongRtn", "ShortRtn"])
    for i, feature_name in enumerate(features_list):
        df = pd.DataFrame(index=result_index, columns=stock_mins_time)
        for id in result_index:
            df.loc[id] = mins_res_dict[id][i]
        for g in range(group_num):
            df.loc[f'Group{g}'] = mins_res_dict[f"Group{g}"][i]
        result_dict[feature_name]["mins_result"] = df

    dir_path = os.path.join(params['result_dir'][0], 'mins', context['date'])
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)
    save_path = os.path.join(dir_path, f'fsdk_mins_bt.pkl')
    with open(save_path, "wb") as f:
        pickle.dump(result_dict, f)

    return {'Temp': np.zeros(all_rtn_arr[0].shape)}
