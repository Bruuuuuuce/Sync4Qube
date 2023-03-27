import abc
import pandas as pd
import numpy as np
from typing import Any, List

class BaseTester(abc.ABC):

    @abc.abstractmethod
    def run_strategy_approximate(self, signals: pd.Series, config: Any) -> Any:
        """
        An approximate simulation of strategy results.

        Return the value of the strategy immediately after settlement.
        """
        pass

    @abc.abstractmethod
    def run_strategy_exact(self, signals: pd.Series, config: Any) -> Any:
        """
        An exact simulation of the strategy (considering the amount of money available)
        """
        pass

    @abc.abstractmethod
    def backtest(self, signals: pd.Series, config: Any) -> Any:
        """
        Major endpoint for users
        """
        pass


class BaseReader(abc.ABC):

    @abc.abstractmethod
    def get_instruments(self, start_date, end_date, pool) -> np.ndarray:
        """
        Return a array of (sorted) instrument code that exists from start_date to end_date
        """
        pass

    @abc.abstractmethod
    def get_trading_days(self, start_date, end_date) -> np.ndarray:
        """
        Return a array of (sorted) trading dates from start_date to end_date (including)
        """
        pass

    @abc.abstractmethod
    def get_series(self, field, instrument, start_date, end_date) -> pd.Series:
        """
        Return a pandas series of the field, the index is the dates
        """
        pass

    @abc.abstractmethod
    def get_panel(self, fields, instruments, start_date, end_date) -> pd.DataFrame:
        """
        Return a pandas dataframe of the fields, the index is a multi-index of [datetime, instrument]
        """
        pass

    @abc.abstractmethod
    def get_returns(self, instruments: List[str], price_field: str, 
            start_date: str, end_date: str, n_before: int, n_after: int) -> pd.Series:
        """
        The return is calculated as:

        return[t] = (price[t+n_after] - price[t-n_before]) / price[t-n_before]
        """
        pass

    @abc.abstractmethod
    def get_buy_status(self, instruments: List[str], price_field: str, 
            start_date: str, end_date: str) -> pd.Series:
        """
        Return True when the instrument can be bought at the specified price.
        """
        pass

    @abc.abstractmethod
    def get_sell_status(self, instruments: List[str], price_field: str, 
            start_date: str, end_date: str) -> pd.Series:
        """
        Return True when the instrument can be sold at the specified price.
        """
        pass