from PqiDataSdk import *
from PqiData import *
import pandas as pd
import numpy as np
import multiprocessing as mp
import json
import copy
from multiprocessing import Manager, Pool, Process, Queue
import logging
import pathlib
import sys
import time
from re import A

sys.path.append("..")

logging.basicConfig(level=logging.CRITICAL)
# from utils.utilsPool import *
# from tools.new_config import *
# import tools.new_config as configPy


myconnector = PqiDataSdk(user="zyding", size=2, pool_type="mt")


def add_prev_vwap(data_minute, buy_window):
    ttval_shift = (
        data_minute["TotalTradeValue"]
        .shift(buy_window, fill_value=0)
        .fillna(method="ffill")
    )
    ttvol_shift = (
        data_minute["TotalTradeVolume"]
        .shift(buy_window, fill_value=0)
        .fillna(method="ffill")
    )
    zero_select = (data_minute["TotalTradeVolume"] - ttvol_shift) < 100
    vwap = (data_minute["TotalTradeValue"] - ttval_shift) / (
        data_minute["TotalTradeVolume"] - ttvol_shift
    )
    vwap = vwap.where(~zero_select, np.nan)
    abnormal_select = np.abs(vwap / data_minute["mp_close"] - 1) > 0.2
    vwap = vwap.where(~abnormal_select, np.nan)
    vwap = vwap.fillna(data_minute["mp_close"]).fillna(method="ffill")
    data_minute[f"PrevVwap{buy_window}"] = vwap
    return data_minute


def add_wp(depth_data):
    """增加wp相关参数"""
    depth_data["wp"] = (
        depth_data["av1"] * depth_data["bp1"] +
        depth_data["bv1"] * depth_data["ap1"]
    ) / (depth_data["av1"] + depth_data["bv1"])
    depth_data["wp"] = (
        depth_data["wp"].replace([0, np.inf], np.nan).fillna(depth_data["mp"])
    )
    wp_twap = depth_data[["minute_index", "wp"]].groupby("minute_index")[
        "wp"].mean()
    wp_twap = (
        depth_data[["minute_index", "wp"]]
        .groupby("minute_index")["wp"]
        .mean()
        .rename("WpTwap")
    )
    wp_open = (
        depth_data[["minute_index", "wp"]]
        .groupby("minute_index")["wp"]
        .first()
        .rename("WpOpen")
    )
    wp_close = (
        depth_data[["minute_index", "wp"]]
        .groupby("minute_index")["wp"]
        .last()
        .rename("WpClose")
    )
    wp_high = (
        depth_data[["minute_index", "wp"]]
        .groupby("minute_index")["wp"]
        .max()
        .rename("WpHigh")
    )
    wp_low = (
        depth_data[["minute_index", "wp"]]
        .groupby("minute_index")["wp"]
        .min()
        .rename("WpLow")
    )
    return pd.concat([wp_twap, wp_open, wp_close, wp_high, wp_low], axis=1)


def add_extend_wap(depth_data):
    """增加wap_twap wbp_twap cum_twap cum_vwap"""
    av_cols = ["av1", "av2", "av3", "av4",
               "av5", "av6", "av7", "av8", "av9", "avA"]
    ap_cols = ["ap1", "ap2", "ap3", "ap4",
               "ap5", "ap6", "ap7", "ap8", "ap9", "apA"]
    bv_cols = ["bv1", "bv2", "bv3", "bv4",
               "bv5", "bv6", "bv7", "bv8", "bv9", "bvA"]
    bp_cols = ["bp1", "bp2", "bp3", "bp4",
               "bp5", "bp6", "bp7", "bp8", "bp9", "bpA"]

    depth_data["wap"] = (
        (
            np.sum(depth_data[av_cols].values *
                   depth_data[ap_cols].values, axis=1)
            / depth_data[av_cols].sum(axis=1)
        )
        .round(6)
        .replace([0, np.inf], np.nan)
        .fillna(depth_data["mp"])
    )
    depth_data["wbp"] = (
        (
            np.sum(depth_data[bv_cols].values *
                   depth_data[bp_cols].values, axis=1)
            / depth_data[bv_cols].sum(axis=1)
        )
        .round(6)
        .replace([0, np.inf], np.nan)
        .fillna(depth_data["mp"])
    )
    wap_twap = (
        depth_data[["minute_index", "wap"]]
        .groupby("minute_index")["wap"]
        .mean()
        .rename("WapTwap")
    )
    wbp_twap = (
        depth_data[["minute_index", "wbp"]]
        .groupby("minute_index")["wbp"]
        .mean()
        .rename("WbpTwap")
    )

    depth_data["_mp_mean"] = depth_data["mp"].expanding().mean()
    cum_twap = (
        depth_data[["minute_index", "_mp_mean"]]
        .groupby("minute_index")["_mp_mean"]
        .mean()
        .rename("CumTwap")
    )

    depth_data["_vwap"] = (
        (depth_data["TotalTradeValue"] / depth_data["TotalTradeVolume"])
    ).replace([0, np.inf], np.nan)
    cum_vwap = (
        depth_data[["minute_index", "_vwap"]]
        .groupby("minute_index")["_vwap"]
        .mean()
        .rename("CumVwap")
    )

    return pd.concat([wap_twap, wbp_twap, cum_twap, cum_vwap], axis=1)


def minute_data_generate(date, ticker):
    """
    df_norm: DataFrame 三表合一DataFrame
    norm_date: str 日期
    norm_ticker: str 标的
    is_pickle: bool 是否存储
    """
    try:
        # 打分钟戳
        df_norm = myconnector.get_3to1_history(
            tickers=ticker,
            start_date=date,
            end_date=date,
            begin_time=93000,
            end_time=145700,
            source="depth",
        )[ticker][date]

        df_norm.sort_values(
            ["TimeStamp", "recieveMdNum"], ascending=[True, True], inplace=True
        )

        lb = min(df_norm["TradePrice"].dropna())
        ub = max(df_norm["TradePrice"].dropna())
        df_norm["PrcNum"] = round(
            (df_norm["TradePrice"] - lb) / (ub - lb) * 199, 0)
        df_norm["ReferencePrice"] = df_norm["PrcNum"].apply(
            lambda x: round(lb + x * (ub - lb) / 199, 4)
        )
        # cut = df_norm.TotalTradeVolume.dropna().values[-1] / 200
        # df_norm["minute_index"] = df_norm[["TotalTradeVolume"]] // cut
        df_norm["minute_index"] = df_norm["PrcNum"]
        df_norm["minute_index"] = df_norm["minute_index"].fillna(
            method="bfill")
        df_norm.rename(columns={"TradeCount": "TotalTradeCount"}, inplace=True)

        origin_depth = myconnector.get_depth_history(
            tickers=ticker,
            start_date=date,
            end_date=date,
            begin_time=93000,
            end_time=145700,
            source="depth",
            fill_na=None,
        )[ticker][date]

        # depth = myconnector.get_depth_history(
        #     tickers=ticker, start_date=date, end_date=date, begin_time=93000, end_time=145700, source="depth"
        # )[ticker][date]
        # dic_tmp = dict(depth.recieveMdNum)
        # dic_depth_num = {value:key for key,value in dic_tmp.items()}
        av_cols = ["av1", "av2", "av3", "av4",
                   "av5", "av6", "av7", "av8", "av9", "avA"]
        bv_cols = ["bv1", "bv2", "bv3", "bv4",
                   "bv5", "bv6", "bv7", "bv8", "bv9", "bvA"]
        df_norm.loc[df_norm.marker == 1,
                    "avSum"] = df_norm[av_cols].sum(axis=1)
        df_norm.loc[df_norm.marker == 1,
                    "bvSum"] = df_norm[bv_cols].sum(axis=1)
        avSum_begin, bvSum_begin = (
            df_norm.loc[
                (df_norm.TimeStamp == 93000000) & (
                    df_norm.marker == 1), "avSum"
            ].iloc[-1],
            df_norm.loc[
                (df_norm.TimeStamp == 93000000) & (
                    df_norm.marker == 1), "bvSum"
            ].iloc[-1],
        )

        ap1 = origin_depth["ap1"].copy()
        bp1 = origin_depth["bp1"].copy()
        ap1.loc[ap1 == 0] = np.nan
        ap1 = ap1.fillna(bp1)
        bp1.loc[bp1 == 0] = np.nan
        bp1 = bp1.fillna(ap1)
        mp = (ap1 + bp1) / 2
        mp[mp == 0] = np.nan
        mp = mp.fillna(method="ffill")
        origin_depth["mp"] = mp
        origin_depth = origin_depth.loc[
            (origin_depth.TimeStamp >= 93000000) & (
                origin_depth.TimeStamp <= 93100000)
        ]
        origin_depth = origin_depth.reset_index(
            drop=True).sort_values(["TimeStamp"])
        if len(origin_depth) > 0:
            first_mp = origin_depth.loc[1, "mp"]
        else:
            first_mp = np.nan
        first_mp = np.nan if first_mp == 0 else first_mp

        # depth 和 trade 数据
        depth_data = (df_norm[df_norm["marker"] == 1]).copy()
        depth_data.reset_index(inplace=True)
        trade_data = (df_norm[df_norm["marker"] == 3]).copy()
        try:
            trade_data = trade_data[trade_data["ExecType"] == 1]
        except:
            trade_data[
                [
                    "OrderIndex",
                    "OrderKind",
                    "TimeStamp",
                    "OrderPrice",
                    "Side",
                    "OrderVolume",
                    "TradeVolume",
                    "BuyIndex",
                    "SellIndex",
                    "TradePrice",
                ]
            ] = ""
        preclose = depth_data.loc[1, "PreClose"]
        trade_data.reset_index(inplace=True)

        # mp计算，先排除涨跌停，然后fill掉mp为0的部分
        # dataserver 4740~4800 的盘口依然在变，不是很河里，fill了一下
        # 两个 flag 分别是涨停跌停
        ap1 = depth_data["ap1"].copy()
        ap1[4739:4799] = np.nan
        bp1 = depth_data["bp1"].copy()
        bp1[4739:4799] = np.nan

        up_limit = np.where((ap1 == 0) & (bp1 != 0), 1, 0)
        # up_limit = (ap1 == 0)
        down_limit = np.where((bp1 == 0) & (ap1 != 0), 1, 0)

        ap1.loc[ap1 == 0] = np.nan
        ap1 = ap1.fillna(bp1)
        bp1.loc[bp1 == 0] = np.nan
        bp1 = bp1.fillna(ap1)
        mp = (ap1 + bp1) / 2
        mp[mp == 0] = np.nan
        mp = mp.fillna(method="ffill")
        depth_data.loc[:, "mp"] = mp
        depth_data.loc[:, "ap1"] = ap1
        depth_data.loc[:, "bp1"] = bp1
        depth_data.loc[:, "up_limit"] = up_limit
        depth_data.loc[:, "down_limit"] = down_limit
        # mp 至今最高价最低价
        mp_cummax = mp.cummax()
        mp_cummin = mp.cummin()
        depth_data.loc[:, "mp_cummax"] = mp_cummax
        depth_data.loc[:, "mp_cummin"] = mp_cummin
        # mp 移动距离
        mp_diff = mp.diff()
        depth_data.loc[:, "mp_diff"] = mp_diff
        # depth 分钟数据生成

        depth_data["TradeVolume"] = depth_data.TotalTradeVolume.diff()
        depth_data["TradeValue"] = depth_data.TotalTradeValue.diff()
        depth_data["TradeCount"] = depth_data.TotalTradeCount.diff()
        depth_data["TimeCount"] = 1
        depth_data["ticker_vwap"] = depth_data["TradeValue"] / \
            depth_data["TradeVolume"]
        depth_data["minute_index"] = round(
            (depth_data["ticker_vwap"] - lb) / (ub - lb) * 199
        )
        depth_data["minute_index"] = depth_data["minute_index"].fillna(
            method="bfill")

        # depth_data["minute_index"] =depth_data["minute_index"].fillna(0)

        depth_data_minute = (
            depth_data[
                ["TradeVolume", "TradeValue", "TradeCount",
                    "minute_index", "TimeCount"]
            ]
            .groupby("minute_index")
            .sum()
        )
        depth_data_minute["TotalTradeVolume"] = depth_data_minute.TradeVolume.cumsum(
        )
        depth_data_minute["TotalTradeValue"] = depth_data_minute.TradeValue.cumsum()
        depth_data_minute["TotalTradeCount"] = depth_data_minute.TradeCount.cumsum()
        depth_data_minute["TotalTimeCount"] = depth_data_minute.TimeCount.cumsum()

        # 增加PdBuyVolDiff PdSellVolDiff
        depth_data["avSum_diff"] = depth_data["avSum"].diff()
        depth_data["bvSum_diff"] = depth_data["bvSum"].diff()

        depth_data.loc[depth_data.TimeStamp == 93003000.000, "avSum_diff"] = (
            depth_data.loc[depth_data.TimeStamp ==
                           93003000.000, "avSum"] - avSum_begin
        )
        depth_data.loc[depth_data.TimeStamp == 93003000.000, "bvSum_diff"] = (
            depth_data.loc[depth_data.TimeStamp ==
                           93003000.000, "bvSum"] - bvSum_begin
        )
        depth_data_minute["PdBuyVolDiff"] = (
            depth_data[["bvSum_diff", "minute_index"]]
            .groupby("minute_index")
            .sum()
            .rename(columns={"bvSum_diff": "PdBuyVolDiff"})
        )
        depth_data_minute["PdSellVolDiff"] = (
            depth_data[["avSum_diff", "minute_index"]]
            .groupby("minute_index")
            .sum()
            .rename(columns={"avSum_diff": "PdSellVolDiff"})
        )

        # 分钟数据拼入高低开收
        depth_data_minute["mp_high"] = (
            depth_data[["minute_index", "mp"]].groupby("minute_index")[
                "mp"].max()
        )
        depth_data_minute["mp_low"] = (
            depth_data[["minute_index", "mp"]].groupby("minute_index")[
                "mp"].min()
        )
        depth_data_minute["mp_open"] = (
            depth_data[["minute_index", "mp"]].groupby("minute_index")[
                "mp"].first()
        )
        depth_data_minute["mp_close"] = (
            depth_data[["minute_index", "mp"]].groupby("minute_index")[
                "mp"].last()
        )
        depth_data_minute["up_limit"] = (
            depth_data[["minute_index", "up_limit"]]
            .groupby("minute_index")["up_limit"]
            .max()
        )
        depth_data_minute["down_limit"] = (
            depth_data[["minute_index", "down_limit"]]
            .groupby("minute_index")["down_limit"]
            .max()
        )

        # 计算Mpdiff
        depth_data_minute["MpDiff"] = depth_data_minute["mp_close"].diff()
        if not np.isnan(first_mp):
            depth_data_minute["MpDiff"].iloc[0] = (
                depth_data_minute["mp_close"].iloc[0] - first_mp
            )
        else:
            depth_data_minute["MpDiff"].iloc[0] = 0

        # 根据 tick 数据进行主动量估计
        ## vwap, last_ap1, last_bp1
        tick_value_diff = depth_data["TotalTradeValue"].diff()
        tick_vol_diff = depth_data["TotalTradeVolume"].diff()
        tick_vwap = tick_value_diff / tick_vol_diff
        tick_vwap[tick_vol_diff == 0] = np.nan
        tick_vwap.fillna(depth_data["mp"], inplace=True)
        last_ap1 = depth_data["ap1"].shift()
        last_bp1 = depth_data["bp1"].shift()
        tick_value_diff[0] = depth_data["TotalTradeValue"][0]
        tick_vol_diff[0] = depth_data["TotalTradeVolume"][0]
        # 这里fill是为了将量等量地分到主买主卖(ap、bp 异常或集合竞价)
        last_ap1.fillna(depth_data["mp"] + 0.01, inplace=True)
        last_bp1.fillna(depth_data["mp"] - 0.01, inplace=True)
        act_sell_ratio = (last_ap1 - tick_vwap) / (last_ap1 - last_bp1)
        # 涨跌停情况下主动量的估计
        act_sell_ratio[(ap1 == 0) & (bp1 != 0)] = 0
        act_sell_ratio[(bp1 == 0) & (ap1 != 0)] = 1
        act_sell_ratio[act_sell_ratio < 0] = 0
        act_sell_ratio[act_sell_ratio > 1] = 1
        depth_data["act_sell_value_depth"] = act_sell_ratio * tick_value_diff
        depth_data["act_sell_volume_depth"] = act_sell_ratio * tick_vol_diff
        depth_data["act_buy_value_depth"] = (
            1 - act_sell_ratio) * tick_value_diff
        depth_data["act_buy_volume_depth"] = (
            1 - act_sell_ratio) * tick_vol_diff

        depth_data_minute["ActSellValDepth"] = (
            depth_data[["minute_index", "act_sell_value_depth"]]
            .groupby("minute_index")["act_sell_value_depth"]
            .sum()
        )

        depth_data_minute["ActSellVolDepth"] = (
            depth_data[["minute_index", "act_sell_volume_depth"]]
            .groupby("minute_index")["act_sell_volume_depth"]
            .sum()
        )
        depth_data_minute["ActBuyValDepth"] = (
            depth_data[["minute_index", "act_buy_value_depth"]]
            .groupby("minute_index")["act_buy_value_depth"]
            .sum()
        )
        depth_data_minute["ActBuyVolDepth"] = (
            depth_data[["minute_index", "act_buy_volume_depth"]]
            .groupby("minute_index")["act_buy_volume_depth"]
            .sum()
        )

        trade_data = trade_data[
            [
                "TimeStamp",
                "BuyIndex",
                "TradePrice",
                "SellIndex",
                "TradeVolume",
                "minute_index",
            ]
        ]

        # 主动买vol、val、count
        act_buy_trade = (
            trade_data[trade_data["BuyIndex"] > trade_data["SellIndex"]]
        ).copy()
        if act_buy_trade.shape[0] > 0:
            # 全天有主动买
            act_buy_trade.loc[:, "TradeValue"] = (
                act_buy_trade["TradePrice"] * act_buy_trade["TradeVolume"]
            )
            # 不做 groupby 的 count
            act_buy_trade.loc[:, "no_groupby_actbuy_count"] = 1

            act_buy_trade["unique_time_ID"] = act_buy_trade["minute_index"].astype(
                str
            ) + act_buy_trade["BuyIndex"].astype(int).astype(str)

            act_buy_info = act_buy_trade.groupby("unique_time_ID")[
                ["TradeValue", "no_groupby_actbuy_count", "TradeVolume"]
            ].sum()

            act_buy_time = act_buy_trade.groupby("unique_time_ID")[
                ["minute_index"]
            ].first()
            act_buy_info = pd.concat([act_buy_info, act_buy_time], axis=1)
            act_buy_info.sort_values("minute_index", inplace=True)
            act_buy_info.loc[:, "groupby_actbuy_count"] = 1
            act_buy_info_sum = act_buy_info.groupby("minute_index")[
                [
                    "TradeValue",
                    "no_groupby_actbuy_count",
                    "TradeVolume",
                    "groupby_actbuy_count",
                ]
            ].sum()

            act_buy_info_sum.rename(
                columns={"TradeVolume": "act_buy_vol"}, inplace=True
            )
            act_buy_info_sum.rename(
                columns={"TradeValue": "act_buy_val"}, inplace=True)
            act_buy_info_sum.rename(
                columns={"no_groupby_actbuy_count": "no_groupby_act_buy_count"},
                inplace=True,
            )
            act_buy_info_sum.rename(
                columns={"groupby_actbuy_count": "groupby_act_buy_count"}, inplace=True
            )
        else:
            act_buy_info_sum = copy.deepcopy(act_buy_trade)
            act_buy_info_sum.loc[:, "TradeValue"] = (
                act_buy_info_sum["TradePrice"] *
                act_buy_info_sum["TradeVolume"]
            )
            act_buy_info_sum.insert(0, "no_groupby_actbuy_count", "")
            act_buy_info_sum.insert(0, "groupby_actbuy_count", "")

            act_buy_info_sum.rename(
                columns={"TradeVolume": "act_buy_vol"}, inplace=True
            )
            act_buy_info_sum.rename(
                columns={"TradeValue": "act_buy_val"}, inplace=True)
            act_buy_info_sum.rename(
                columns={"no_groupby_actbuy_count": "no_groupby_act_buy_count"},
                inplace=True,
            )
            act_buy_info_sum.rename(
                columns={"groupby_actbuy_count": "groupby_act_buy_count"}, inplace=True
            )

            act_buy_info_sum = act_buy_info_sum[
                [
                    "act_buy_vol",
                    "act_buy_val",
                    "no_groupby_act_buy_count",
                    "groupby_act_buy_count",
                ]
            ]

        depth_data_minute = pd.concat(
            [depth_data_minute, act_buy_info_sum], axis=1)

        # 主动卖vol、val、count
        act_sell_trade = (
            trade_data[trade_data["SellIndex"] > trade_data["BuyIndex"]]
        ).copy()
        if act_sell_trade.shape[0] > 0:
            act_sell_trade.loc[:, "TradeValue"] = (
                act_sell_trade["TradePrice"] * act_sell_trade["TradeVolume"]
            )

            # 不做 groupby count
            act_sell_trade.loc[:, "no_groupby_actsell_count"] = 1

            act_sell_trade["unique_time_ID"] = act_sell_trade["minute_index"].astype(
                str
            ) + act_sell_trade["SellIndex"].astype(int).astype(str)

            act_sell_info = act_sell_trade.groupby("unique_time_ID")[
                ["TradeValue", "no_groupby_actsell_count", "TradeVolume"]
            ].sum()
            act_sell_time = act_sell_trade.groupby("unique_time_ID")[
                ["minute_index"]
            ].first()

            act_sell_info = pd.concat([act_sell_info, act_sell_time], axis=1)
            act_sell_info.sort_values("minute_index", inplace=True)
            act_sell_info.loc[:, "groupby_actsell_count"] = 1

            act_sell_info_sum = act_sell_info.groupby("minute_index")[
                [
                    "TradeValue",
                    "no_groupby_actsell_count",
                    "TradeVolume",
                    "groupby_actsell_count",
                ]
            ].sum()
            # act_sell_info_index = list(act_sell_info_sum.index)
            # act_sell_info_index = np.array(act_sell_info_index) + 1
            # act_sell_info_index[np.where(act_sell_info_index == 960)] = 1000
            # act_sell_info_index[np.where(act_sell_info_index == 1060)] = 1100
            # act_sell_info_index[np.where(act_sell_info_index == 1360)] = 1400
            # act_sell_info_index[np.where(act_sell_info_index == 1460)] = 1500
            # act_sell_info_index[np.where(act_sell_info_index == 1501)] = 1500
            # act_sell_info_sum = act_sell_info_sum.set_index(act_sell_info_index)
            # act_sell_info_sum = act_sell_info_sum.loc[
            #     ~act_sell_info_sum.index.duplicated(keep="first")
            # ]
            act_sell_info_sum.rename(
                columns={"TradeVolume": "act_sell_vol"}, inplace=True
            )
            act_sell_info_sum.rename(
                columns={"TradeValue": "act_sell_val"}, inplace=True
            )
            act_sell_info_sum.rename(
                columns={"no_groupby_actsell_count": "no_groupby_act_sell_count"},
                inplace=True,
            )
            act_sell_info_sum.rename(
                columns={"groupby_actsell_count": "groupby_act_sell_count"},
                inplace=True,
            )

        else:
            act_sell_info_sum = act_sell_trade.copy()
            act_sell_info_sum.loc[:, "TradeValue"] = (
                act_sell_info_sum["TradePrice"] *
                act_sell_info_sum["TradeVolume"]
            )
            act_sell_info_sum.insert(0, "no_groupby_actsell_count", "")
            act_sell_info_sum.insert(0, "groupby_actsell_count", "")

            act_sell_info_sum.rename(
                columns={"TradeVolume": "act_sell_vol"}, inplace=True
            )
            act_sell_info_sum.rename(
                columns={"TradeValue": "act_sell_val"}, inplace=True
            )
            act_sell_info_sum.rename(
                columns={"no_groupby_actsell_count": "no_groupby_act_sell_count"},
                inplace=True,
            )
            act_sell_info_sum.rename(
                columns={"groupby_actsell_count": "groupby_act_sell_count"},
                inplace=True,
            )

            act_sell_info_sum = act_sell_info_sum[
                [
                    "act_sell_vol",
                    "act_sell_val",
                    "no_groupby_act_sell_count",
                    "groupby_act_sell_count",
                ]
            ]

        data_minute = pd.concat([depth_data_minute, act_sell_info_sum], axis=1)
        # df_std = pd.DataFrame(index=range(0,200))
        # data_minute = pd.concat([df_std,data_minute],axis=1)

        data_minute = data_minute.reset_index()
        data_minute.rename(
            columns={
                "minute_index": "PrcNum",
                "act_buy_vol": "TotalActBuyVol",
                "act_buy_val": "TotalActBuyVal",
                "act_sell_vol": "TotalActSellVol",
                "act_sell_val": "TotalActSellVal",
            },
            inplace=True,
        )

        data_minute["ReferencePrice"] = data_minute["PrcNum"].apply(
            lambda x: round(lb + x * (ub - lb) / 199, 4)
        )

        # 讲关于价格字段填充为nan， 其余关于金额和量的填充为0
        cols = [
            "PrcNum",
            "ReferencePrice",
            "TradeCount",
            "TotalTradeCount",
            "TradeVolume",
            "TotalTradeVolume",
            "TradeValue",
            "TotalTradeValue",
            "TimeCount",
            "TotalTimeCount",
            "TotalActBuyVal",
            "TotalActBuyVol",
            "TotalActSellVal",
            "TotalActSellVol",
        ]
        data_minute = data_minute.fillna(0)
        data_minute = data_minute[cols]
        # print(data_minute)
        pathlib.Path(f"{PATH}{date}/").mkdir(exist_ok=True)
        data_minute.to_pickle(f"{PATH}{date}/{ticker}.pkl")
    except Exception as e:
        print(e)


def run(date):
    #myconnector = PqiDataSdk(user="zyding", size=2, pool_type="mt")
    print('Creating pool ...')
    pool = Pool(32)
    tickers = myconnector.get_ticker_list(date=date)[:20]
    for ticker in tickers:
        print('Applying process ...')
        pool.apply_async(
            minute_data_generate,
            args=(
                date,
                ticker,
            ),
        )
        time.sleep(0.1)
    pool.close()
    pool.join()


if __name__ == "__main__":
    PATH = './'
    ticker = "600519"
    date = "20200225"
    minute_data_generate(date, ticker)
    #run(date)
