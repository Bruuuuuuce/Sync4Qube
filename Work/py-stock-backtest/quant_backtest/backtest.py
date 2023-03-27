from datetime import datetime
import os
import warnings
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .utils.numba_utils import run_strategy_cross_sectional_numba
from common_utils.utils import timeit

def reorganize_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['object_id'], axis=1)
    df['trade_dt'] = df['trade_dt'].map(lambda t: datetime.strptime(str(t), '%Y%m%d'))
    df.index = pd.MultiIndex.from_arrays([df['trade_dt'], df['s_info_windcode']],
                                         names=['datetime', 'instrument'])
    df = df.drop(['s_info_windcode', 'trade_dt'], axis=1)
    df = df.sort_index()
    return df

def nested_log(info: dict, log: dict, prefix: str = ''):
    prefix = '' if prefix == '' else f'{prefix}/'
    for key, val in info.items():
        if isinstance(val, dict):
            nested_log(val, log, f'{prefix}{key}')
        else:
            log[f'{prefix}{key}'] = val


class BackTest:

    def __init__(self, start_date, end_date, data_path, 
            holding_periods: List[int], n_groups: int,
            price='close', cache_dir='cache') -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.holding_periods = holding_periods
        self.n_groups = n_groups
        self.data_path = data_path
        self.price = price
        self.data = None
        self.cache_dir = cache_dir

    def setup_data(self):
        cache_path = os.path.join(self.cache_dir, f'backtest_{self.start_date}_{self.end_date}.pkl')
        if os.path.exists(cache_path):
            self.data = pd.read_pickle(cache_path)
            return
        df = pd.read_csv(self.data_path)
        df = reorganize_raw_df(df)
        df = df[['s_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 
                 's_dq_adjclose', 's_dq_volume', 's_dq_avgprice', 's_dq_tradestatuscode',
                 'S_DQ_LIMIT'.lower(), 'S_DQ_STOPPING'.lower(), 'S_DQ_CLOSE'.lower()]]
        df.columns = ['open', 'high', 'low', 'close', 'vol', 'vwap', 'statuscode',
                      'buy_limit_price', 'sell_limit_price', 'real_close']
        # drop unnormal data (volume = 0)
        df = df[df['vol'] > 0]
        # df = df[self.price]
        # add missing values
        df = df.unstack().stack(dropna=False)
        df = df.sort_index()
        df = df.loc[~df.groupby('instrument').ffill().isnull()[self.price]]
        df = df.loc[~df.groupby('instrument').bfill().isnull()[self.price]]
        df = df.loc[self.start_date:self.end_date]
        self.data = df
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.data.to_pickle(cache_path)

    def shift_date(self, alpha: pd.Series, n_shift: int = 0) -> pd.Series:
        return alpha.groupby('instrument').shift(n_shift)

    def cal_strategy_metrics(self, values: np.ndarray, benchmark: np.ndarray, 
            start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
        # return related
        n_days = (end_date - start_date).days
        annual_return = values[-1] ** (365/n_days)
        benchmark_return = benchmark[-1] ** (365/n_days)
        # drawdown
        max_drawdown = 1 - (values / np.maximum.accumulate(values)).min()
        # sharpe
        daily_return = np.diff(values) / values[:-1]
        daily_sharpe = np.mean(daily_return) / np.std(daily_return)
        annual_sharpe = daily_sharpe * np.sqrt(250)

        return {
            'annual_return': annual_return,
            'benchmark_return': benchmark_return,
            'excess_return': annual_return - benchmark_return,
            'max_drawdown': max_drawdown,
            'sharpe': annual_sharpe
        }

    def alpha_backtest(self, alpha: pd.Series, alpha_shifted=True, 
            plot=False, save_dir: str = 'outputs', config: dict = {},
            numba=True) -> dict:
        if not alpha_shifted:
            alpha = self.shift_date(alpha, 1)
        alpha = alpha.dropna()
        results = {}
        for holding_period in self.holding_periods:
            stock_return = self.get_future_return(holding_period, normalize=True)
            ic_results = self.alpha_ic_test(alpha, stock_return)
            cfg = {
                'holding_period': holding_period
            }
            cfg.update(config)
            strategy_results = self.run_strategy(alpha, alpha_shifted=True, future_return=None,
                                                cfg=cfg, plot=plot,
                                                save_path=os.path.join(save_dir, f'holding-{holding_period}.png'),
                                                numba=numba)
            info = {}
            info.update(ic_results)
            strategy_metrics = self.cal_strategy_metrics(
                strategy_results['value_with_fee'],
                strategy_results['avg_value'],
                start_date=strategy_results['trading_days'].min(),
                end_date=strategy_results['trading_days'].max())
            nested_log(strategy_metrics, info, 'w_fee')
            strategy_metrics = self.cal_strategy_metrics(
                strategy_results['value_no_fee'],
                strategy_results['avg_value'],
                start_date=strategy_results['trading_days'].min(),
                end_date=strategy_results['trading_days'].max())
            nested_log(strategy_metrics, info, 'wo_fee')
            info['turnover'] = strategy_results['turnover_rate'].mean()
            results[holding_period] = info
        return results

    @timeit
    def get_future_return(self, n_days_ahead, normalize, dropna=False):
        if self.data is None:
            self.setup_data()
        data = self.data
        if dropna:
            data = data.dropna()
        future_price = data.groupby('instrument')[self.price].shift(-n_days_ahead)
        future_return = future_price / data[self.price]
        # future_return = future_return.dropna()
        if normalize:
            future_return = future_return ** (1/n_days_ahead)
        future_return -= 1
        return future_return

    @timeit
    def group_test(self, alpha: pd.Series, future_return: pd.Series, plot: bool, save_path: str):
        warnings.warn('group_test() is deprecated. Use run_strategy() instead.')
        # add random noise to alpha so that each group has stocks when the alpha is discrete
        alpha += np.random.normal(0, 1e-6, alpha.shape[0])
        df = pd.concat([alpha, future_return], axis=1, join='inner')
        df = df.sort_index()
        df.columns = ['alpha', 'return']
        df['group'] = df.groupby('datetime')['alpha'].rank(pct=True
            ).map(lambda x: int(self.n_groups * x) if x != 1 else self.n_groups-1)
        top_avg_return = df[df['group'] == self.n_groups-1].groupby('datetime')['return'].mean()
        avg_return = df.groupby('datetime')['return'].mean()
        df = pd.concat([top_avg_return, avg_return], axis=1)
        df.columns = ['top', 'avg']
        cum_value = (df + 1).shift(1).fillna(1).cumprod()
        info = {}
        begin_date = cum_value.index.min()
        end_date = cum_value.index.max()
        n_days = (end_date - begin_date).days
        info['top_return'] = cum_value['top'].iloc[-1] ** (365/n_days) - 1
        info['avg_return'] = cum_value['avg'].iloc[-1] ** (365/n_days) - 1
        info['excess'] = info['top_return'] - info['avg_return']
        if plot:
            # do plot
            cum_value.plot()
            parent_dir = os.path.dirname(save_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            plt.savefig(save_path, dpi=1000)
            plt.close()
            print(f'Plot saved to {save_path}.')
        return info

    @timeit
    def alpha_ic_test(self, alpha: pd.Series, future_return: pd.Series):
        df = pd.concat([alpha, future_return], axis=1, join='inner')
        df.columns = ['alpha', 'return']
        ic = df.groupby('datetime').corr().groupby(level=1).mean().loc['alpha', 'return']
        rankic = df.groupby('datetime').corr(method='spearman').groupby(level=1).mean().loc['alpha', 'return']
        return {'IC': ic, 'RankIC': rankic}

    @staticmethod
    def run_strategy_static_ordinary(alpha: np.ndarray, today_return: np.ndarray, 
        can_buy: np.ndarray, can_sell: np.ndarray, n_trading_days: int,
        pool_size: int, holding_period: int, buy_fee_ratio: float, 
        sell_fee_ratio: float, holding_percentage: float):
        """
        Run simple strategy backtests. alpha series need to be shifted.

        Consider:
        - transaction fee (assume deal at close price, and do not need to adjust portfolio due to transaction fee)
        - buy and sell limit

        Do not consider:
        - division rounding problem
        """

        portfolio_weights = np.zeros((n_trading_days, holding_period, pool_size))
        # the portfolio value is calculated after all transaction of the day (i.e., after the market is closed)
        raw_values = np.ones(n_trading_days) # before fee
        net_values = np.ones(n_trading_days) # after fee
        change_over = np.zeros(n_trading_days) # 单边换手

        # the main loop
        for i in range(n_trading_days):
            # update portfolio weight
            group_id = i % holding_period
            # import pdb; pdb.set_trace()
            alp = alpha[i]
            alp_argindex = alp.argsort()
            alp_argindex = alp_argindex[~np.isnan(alp[alp_argindex])]
            alp_select = alp_argindex[-int(holding_percentage*len(alp_argindex)):]
            # consider buy and sell limits
            # consider sell limits
            if i > 0:
                # need to sell some portion
                cannot_sell = 1-np.nan_to_num(can_sell[i], 0)   # 可能出现停牌等导致中间数据确实，不能卖出
                cannot_sell *= portfolio_weights[i-1, group_id]
            else:
                cannot_sell = np.zeros_like(portfolio_weights[i, group_id])
            # import pdb; pdb.set_trace()
            cannot_sell_portion = cannot_sell.sum()
            for instr_index in alp_select:
                if cannot_sell[instr_index] == 0:
                    portfolio_weights[i, group_id, instr_index] = 1
            portfolio_weights[i, group_id] *= np.nan_to_num(can_buy[i], 0)
            portfolio_weights[i, group_id] *= (1/holding_period - cannot_sell_portion) \
                 / portfolio_weights[i, group_id].sum()
            portfolio_weights[i, group_id] += cannot_sell
            # import pdb; pdb.set_trace()
            assert np.abs(portfolio_weights[i, group_id].sum() - 1/holding_period) < 1e-10
            # calculate transaction fee
            weight_diff = portfolio_weights[i, group_id] - portfolio_weights[i-1, group_id] \
                if i > 0 else portfolio_weights[i, group_id]
            change_over[i] = np.abs(weight_diff).sum() / 2
            buy_portion = weight_diff[weight_diff > 0].sum()
            sell_portion = -weight_diff[weight_diff < 0].sum()
            # update stock return to portfolio
            # copy other group weights
            for j in range(holding_period):
                if j != group_id:
                    portfolio_weights[i, j] = portfolio_weights[i-1, j]
            yesterday_weight = portfolio_weights[i-1].sum(axis=0) if i > 0 else np.zeros(pool_size)
            assert np.abs(yesterday_weight.sum() - min(1, i/holding_period)) < 1e-10
            assert np.logical_or(weight_diff == 0, ~np.isnan(today_return[i])).all()
            today_ret = np.nansum(yesterday_weight * today_return[i])
            raw_values[i] = (raw_values[i-1] if i > 0 else 1) * (1 + today_ret)
            net_values[i] = (net_values[i-1] if i > 0 else 1) * (1 + today_ret)
            # reduct fee
            buy_fee = net_values[i] * buy_portion * buy_fee_ratio
            sell_fee = net_values[i] * sell_portion * sell_fee_ratio
            net_values[i] -= (buy_fee + sell_fee)
        # record results
        portfolio_weights = portfolio_weights.sum(axis=1)
        assert np.allclose(portfolio_weights.sum(axis=-1), 
                           np.minimum(1, (np.arange(n_trading_days)+1) / holding_period), 
                           1e-10, 1e-12, False)
        # avg_value = (np.nanmean(today_return, axis=1)+1).cumprod()
        benchmark = np.cumprod(np.nan_to_num(today_return)+1, axis=0).mean(axis=1)
        return net_values, raw_values, benchmark, change_over, portfolio_weights

    @timeit
    def run_strategy(self, alpha: pd.Series, alpha_shifted: bool, future_return: pd.Series, cfg: dict, 
            plot: bool, save_path: str, numba: bool = True):
        start_date = cfg['start_date'] if 'start_date' in cfg else self.start_date
        end_date = cfg['end_date'] if 'end_date' in cfg else self.end_date
        if not alpha_shifted:
            alpha = self.shift_date(alpha, 1)
        if future_return is None:
            future_return = self.get_future_return(1, False, dropna=True)
        today_return = future_return.groupby('instrument').shift(1)  # return of T-1 to T, in terms of close price
        today_return = today_return.loc[self.start_date:self.end_date]
        today_return = today_return.dropna()
        start_date = max(today_return.index.get_level_values(0).min().strftime('%Y-%m-%d'), start_date)
        end_date = min(today_return.index.get_level_values(0).max().strftime('%Y-%m-%d'), end_date)
        if start_date > self.start_date or end_date < self.end_date:
            warnings.warn(f'Strategy backtest timespan: [{start_date}, {end_date}]')
        alpha = alpha.loc[start_date:end_date]
        today_return = today_return.loc[start_date:end_date]

        instruments = today_return.index.get_level_values('instrument').unique().sort_values()
        instrument_indexer = {code: index for index, code in enumerate(instruments)}
        pool_size = len(instruments)
        trading_days = today_return.index.get_level_values('datetime').unique()
        trading_days = trading_days.sort_values()
        n_trading_days = len(trading_days)
        # for merge (alpha)
        alpha.name = 'alpha'
        alpha = pd.merge(today_return, alpha, how='left', left_index=True, right_index=True)['alpha']
        alpha = alpha.unstack().loc[trading_days, instruments]
        assert (alpha.index == trading_days).all()
        assert (alpha.columns == instruments).all()

        today_return = today_return.unstack().loc[trading_days, instruments]
        assert (today_return.columns == instruments).all()
        assert (today_return.index == trading_days).all()

        # pre calculate buy and sell limits
        can_buy = self.data['real_close'] < self.data['buy_limit_price']
        can_buy &= self.data['statuscode'] >= -1
        can_buy = can_buy.unstack()[instruments].loc[trading_days]
        assert (can_buy.columns == instruments).all()
        assert (can_buy.index == trading_days).all()

        can_sell = self.data['real_close'] > self.data['sell_limit_price']
        can_sell &= self.data['statuscode'] >= -1
        can_sell = can_sell.unstack()[instruments].loc[trading_days]
        assert (can_sell.columns == instruments).all()
        assert (can_sell.index == trading_days).all()

        # make everything into numpy for numba
        alpha = alpha.to_numpy()
        today_return = today_return.to_numpy()
        can_sell = can_sell.to_numpy().astype(float)
        can_buy = can_buy.to_numpy().astype(float)

        func = run_strategy_cross_sectional_numba if numba else self.run_strategy_static_ordinary
        net_values, raw_values, avg_value, change_over, portfolio_weights = func(alpha, today_return, can_buy, can_sell, 
            n_trading_days, pool_size, cfg['holding_period'], cfg['buy_fee_ratio'],
            cfg['sell_fee_ratio'], cfg['hold_percentage'])
            
        results = {
            'value_no_fee': raw_values,
            'value_with_fee': net_values,
            'turnover_rate': change_over,
            'portfolio_weights': portfolio_weights,
            'trading_days': trading_days,
            'instruments': instruments,
            'avg_value': avg_value
        }
        if plot:
            plt.plot(results['trading_days'], results['value_with_fee'], label='w/ fee')
            plt.plot(results['trading_days'], results['value_no_fee'], label='w/o fee')
            plt.plot(results['trading_days'], results['avg_value'], label='avg')
            plt.legend()
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, dpi=500)
            plt.close()
        return results