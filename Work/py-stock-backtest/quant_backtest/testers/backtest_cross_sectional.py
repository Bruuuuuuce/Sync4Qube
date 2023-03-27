import logging
from typing import Any, List, Union

import numpy as np

from ..utils.numba_utils import run_strategy_cross_sectional_numba
from ..utils.backtest_utils import prepare_signals, run_strategy_cross_sectional
from ..base import BaseTester, BaseReader
from ..config import CrossSectionalConfig
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from common_utils.utils import create_logger, timeit
import os

class CrossSectionalTester(BaseTester):

    def __init__(self, database_reader: BaseReader) -> None:
        self.reader = database_reader

    def run_strategy_exact(self, signals: Union[pd.Series, np.ndarray], config: Any) -> Any:
        raise NotImplementedError

    def run_strategy_approximate(self, signals: Union[pd.Series, np.ndarray], config: CrossSectionalConfig,
            signals_ready: bool = False, instruments: List[str] = None,
            trading_days: List[str] = None) -> dict:
        create_logger(__name__).info('Preparing for approximate backtest simulation ...')
        start_date = config.start_date
        end_date = config.end_date
        if instruments is None:
            instruments = self.reader.get_instruments(start_date, end_date, config.pool)
        if trading_days is None:
            trading_days = self.reader.get_trading_days(start_date, end_date)

        create_logger(__name__).debug('Starting to prepare input signal for backtest ...')
        if not signals_ready:
            signals = prepare_signals(signals, True, trading_days)
            instruments = signals.columns.to_numpy()
        create_logger(__name__).debug('Input signal is prepared.')

        create_logger(__name__).debug('Preparing daily return and buy sell constraints ...')
        today_return = self.reader.get_returns(instruments, config.price, start_date, end_date, 1, 0)
        today_return = today_return.unstack().loc[trading_days, instruments]
        can_buy = self.reader.get_buy_status(instruments, config.price, start_date, end_date)
        can_sell = self.reader.get_sell_status(instruments, config.price, start_date, end_date)
        can_buy = can_buy.unstack().loc[trading_days, instruments]
        can_sell = can_sell.unstack().loc[trading_days, instruments]

        create_logger(__name__).debug('checking index and column consistency ...')
        assert (signals.index == trading_days).all()
        assert (signals.columns == instruments).all()
        assert (today_return.index == trading_days).all()
        assert (today_return.columns == instruments).all()
        assert (can_buy.index == trading_days).all()
        assert (can_buy.columns == instruments).all()
        assert (can_sell.index == trading_days).all()
        assert (can_sell.columns == instruments).all()
        
        create_logger(__name__).debug('converting to numpy array ...')
        signals = signals.to_numpy()
        today_return = today_return.to_numpy()
        can_buy = can_buy.to_numpy().astype(float)
        can_sell = can_sell.to_numpy().astype(float)

        create_logger(__name__).info('Running approximated backtest with 4 job parallel ...')
        # import pdb; pdb.set_trace()
        outputs = Parallel(n_jobs=4, max_nbytes='1G')(
            delayed(run_strategy_cross_sectional)(signals, today_return, can_buy, can_sell,
                                         len(trading_days), len(instruments),
                                         h_period, config.buy_cost, config.sell_cost,
                                         config.holding_percentage) \
                for h_period in config.holding_periods
        )
        # outputs = [run_strategy_cross_sectional_numba(signals, today_return, can_buy, can_sell,
        #                                  len(trading_days), len(instruments),
        #                                  h_period, config.buy_cost, config.sell_cost,
        #                                  config.holding_percentage) \
        #                 for h_period in config.holding_periods]
        strategy_results = {}
        for h_period, result in zip(config.holding_periods, outputs):
            strategy_results[h_period] = {
                'value_no_fee': result[1],
                'value_with_fee': result[0],
                'turnover_rate': result[3],
                'portfolio_weights': result[4],
                'trading_days': trading_days,
                'instruments': instruments,
                'avg_value': result[2]
            }
        create_logger(__name__).info('Approximated backtest is successful.')
        # net_values, raw_values, avg_value, change_over, portfolio_weights = list(zip(*outputs))
        return strategy_results

    @timeit(level=logging.INFO)
    def _ic_backtest(self, signals: np.ndarray, trading_days: List[str], 
            instruments: List[str], holding_period: int, price_field: str,
            ngroups: int, workers: int = 4):
        start_date = str(min(trading_days)).split('T')[0]
        end_date = str(max(trading_days)).split('T')[0]
        future_return = self.reader.get_returns(instruments, price_field, start_date,
                                                end_date, 0, holding_period)
        future_return = future_return.unstack()
        future_return = future_return.loc[trading_days, instruments]
        assert (future_return.index == trading_days).all()
        assert (future_return.columns == instruments).all()
        future_return = future_return.to_numpy()

        # ic calculation
        def calculate_ic_rankic(x: np.ndarray, y: np.ndarray) -> dict:
            null_mask = np.logical_or(np.isnan(x), np.isnan(y))
            if null_mask.all():
                # all nan
                return {'ic': np.nan, 'rankic': np.nan}
            x = x[~null_mask]
            y = y[~null_mask]
            ic = np.corrcoef(x, y)[0, 1]
            rank_x = np.argsort(np.argsort(x))
            rank_y = np.argsort(np.argsort(y))
            rankic = np.corrcoef(rank_x, rank_y)[0, 1]
            return {'ic': ic, 'rankic': rankic}

        create_logger(__name__).debug('Start calculating IC and Rank IC ...')
        raw_ic_outputs = Parallel(n_jobs=workers)(
            delayed(calculate_ic_rankic)(x, y) for x, y in zip(signals, future_return)
        )
        # raw_ic_outputs = [
        #     calculate_ic_rankic(x, y) for x, y in zip(signals, future_return)
        # ]
        create_logger(__name__).debug('IC calculation is successful.')
        ic_results = pd.DataFrame.from_records(raw_ic_outputs)
        ic_results.index = trading_days
        ic_outputs = {
            'ic': ic_results['ic'].mean(),
            'rankic': ic_results['rankic'].mean(),
            'icir': ic_results['ic'].mean() / ic_results['ic'].std(),
            'rank_icir': ic_results['rankic'].mean() / ic_results['rankic'].std(),
            'ic_value': ic_results['ic'],
            'rankic_value': ic_results['rankic'],
        }

        # grouped tests
        def cal_group_return(x: np.ndarray, y: np.ndarray) -> dict:
            null_mask = np.logical_or(np.isnan(x), np.isnan(y))
            if null_mask.all():
                # all nan
                return {i: np.nan for i in range(ngroups)}
            x = x[~null_mask]
            y = y[~null_mask]
            grouped_y = y[np.argsort(x)]
            group_num = len(y) // ngroups + 1
            group_return = {}
            for i in range(ngroups):
                group_return[i] = np.mean(grouped_y[group_num*i:group_num*(i+1)])
            return group_return
        
        create_logger(__name__).debug(f'Start grouped testing with {ngroups} groups ...')
        raw_group_output = Parallel(n_jobs=workers)(
            delayed(cal_group_return)(x, y) for x, y in zip(signals, future_return)
        )
        create_logger(__name__).debug('Group testing is successful.')
        group_result = pd.DataFrame.from_records(raw_group_output)
        group_result.index = trading_days
        group_output = {
            'group_return': group_result.mean(),
            'group_return_value': group_result
        }

        output = {}
        output.update(ic_outputs)
        output.update(group_output)
        return output

    def ic_backtest(self, signals: Union[pd.Series, np.ndarray],
            config: CrossSectionalConfig, signals_ready: bool = False,
            instruments: List[str] = None, trading_days: List[np.datetime64] = None,
            workers: int = 4):
        """
        Perform IC and grouped testing.
        """

        create_logger(__name__).debug('Starting to prepare input signal for IC backtest ...')
        if not signals_ready:
            signals = prepare_signals(signals, True, trading_days)
            instruments = signals.columns.to_numpy()
        create_logger(__name__).debug('Input signal is prepared.')

        create_logger(__name__).info(f'Starting IC backtest with {workers} job parallel ...')
        outputs = [
            self._ic_backtest(signals, trading_days, instruments, h_period, 
                    config.price, config.ngroups, workers) \
                for h_period in config.holding_periods
        ]

        ic_results = {}
        for h_period, result in zip(config.holding_periods, outputs):
            ic_results[h_period] = {
                'ic': result['ic'],
                'rankic': result['rankic'],
                'icir': result['icir'],
                'rank_icir': result['rank_icir'],
                'ic_value': result['ic_value'],
                'rankic_value': result['rankic_value'],
                'group_return': result['group_return'],
                'group_return_value': result['group_return_value'],
            }

        create_logger(__name__).info('IC backtest is successful.')
        return ic_results

    def backtest(self, signals: pd.Series, config: CrossSectionalConfig, plot: bool = True) -> dict:
        start_date = config.start_date
        end_date = config.end_date
        # instruments = self.reader.get_instruments(start_date, end_date, config.pool)
        trading_days = self.reader.get_trading_days(start_date, end_date)
        # calculate IC/RankIC
        create_logger(__name__).info('Preparing signals for cross-sectional backtest ...')
        signals = prepare_signals(signals, True, trading_days)
        instruments = signals.columns.to_numpy()
        trading_days = signals.index.to_numpy()
        ic_results = self.ic_backtest(signals.to_numpy(), config, True, instruments, trading_days)
        strategy_results = self.run_strategy_approximate(signals, config, 
            signals_ready=True, instruments=instruments, trading_days=trading_days)
        # plotting
        if plot:
            # ic result
            result = {}
            for key in ic_results.keys():
                result[f'h-{key}'] = ic_results[key]['group_return']
            result = pd.DataFrame(result)
            print(result)
            result = ic_results.copy()
            for key in result.keys():
                result[key].pop('ic_value')
                result[key].pop('rankic_value')
                result[key].pop('group_return')
                result[key].pop('group_return_value')
            result = pd.DataFrame.from_dict(result)
            print(result)
            # strategy result
            for holding_period, results in strategy_results.items():
                plt.plot(results['trading_days'], results['value_with_fee'], label='w/ fee')
                plt.plot(results['trading_days'], results['value_no_fee'], label='w/o fee')
                plt.plot(results['trading_days'], results['avg_value'], label='avg')
                plt.legend()
                save_path = f'outputs/test-{holding_period}.png'
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(save_path, dpi=500)
                plt.close()