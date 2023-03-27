import logging
from typing import List, Union
import numpy as np
import pandas as pd

from common_utils.utils import timeit
from ..base import BaseReader
from quant_database_backend.interface_ddb import DDBDatabaseReader


class DDBReader(BaseReader):

    def __init__(self):
        self.backend = DDBDatabaseReader()

    @timeit
    def get_instruments(self, start_date, end_date, pool) -> np.ndarray:
        return self.backend.get_instruments(start_date, end_date, pool)

    @timeit
    def get_trading_days(self, start_date, end_date) -> np.ndarray:
        return self.backend.get_trading_days(start_date, end_date)

    @timeit
    def get_series(self, field, instrument, start_date, end_date, adj) -> pd.Series:
        return self.backend.get_series(field, instrument, start_date, end_date, adj)

    @timeit
    def get_panel(self, fields, instruments, start_date, end_date, adj) -> pd.DataFrame:
        return self.backend.get_panel(fields, instruments, start_date, end_date, adj)

    @timeit(level=logging.INFO)
    def get_returns(self, instruments: Union[List[str], np.ndarray], price_field: str, 
            start_date: str, end_date: str, 
            n_before: int, n_after: int,
            method: str = 'pandas') -> pd.Series:
        if method == 'pandas':
            price = self.backend.get_panel(
                [price_field], instruments, 
                self.backend.shift_date(start_date, -n_before),
                self.backend.shift_date(end_date, n_after), 
                adj=True)[price_field]
            price_after = price.groupby('instrument').shift(-n_after)
            price_before = price.groupby('instrument').shift(n_before)
            ret = price_after/price_before - 1
            return ret.loc[start_date:end_date]
        else:
            raise ValueError(f'method {method} is unrecognized.')

    @timeit(level=logging.INFO)
    def get_buy_status(self, instruments: Union[List[str], np.ndarray], price_field: str, 
            start_date: str, end_date: str) -> pd.Series:
        table = self.backend._get_panel_raw_table(
            [price_field, 'up_limit', 'trade_status'], instruments,
            start_date, end_date, adj=False
        )
        table = table.select(f'{price_field}<up_limit as can_buy,trade_status,datetime,instrument')
        table = table.select('can_buy&&(trade_status==0) as can_buy,datetime,instrument')
        df = table.toDF()
        can_buy = df['can_buy']
        can_buy.index = pd.MultiIndex.from_frame(df[['datetime', 'instrument']])
        return can_buy

    @timeit(level=logging.INFO)
    def get_sell_status(self, instruments: Union[List[str], np.ndarray], price_field: str, 
            start_date: str, end_date: str) -> pd.Series:
        table = self.backend._get_panel_raw_table(
            [price_field, 'down_limit', 'trade_status'], instruments,
            start_date, end_date, adj=False
        )
        table = table.select(f'{price_field}>down_limit as can_sell,trade_status,datetime,instrument')
        table = table.select('can_sell&&(trade_status==0) as can_sell,datetime,instrument')
        df = table.toDF()
        can_sell = df['can_sell']
        can_sell.index = pd.MultiIndex.from_frame(df[['datetime', 'instrument']])
        return can_sell