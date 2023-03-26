
from sklearn.utils import resample
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from functions import mins_functions
from yarl import cache_clear



from PqiDataSdk import *


class generator():
    def __init__(self,):
        pass
    
    @staticmethod
    def calc_mins(tickers,start_date,end_date,mins=30):
        functions = mins_functions
        ds = PqiDataSdk(size=1)
        dates = ds.get_trade_dates(start_date=start_date,
                                    end_date=end_date)
        small_data_df = pd.DataFrame(index=pd.DataFrame(index=tickers,
                        columns=dates).stack(dropna=False).index,
                        columns=list(range(240//mins)))
        feature_names = list(functions.keys())
        data_df = pd.DataFrame(index=small_data_df.stack(dropna=False).index,
                               columns=feature_names)
        for ticker in tickers:
            mins_data = ds.get_mins_history(tickers=ticker,
                                            start_date=start_date,
                                            end_date=end_date)
            for date in dates:
                try:
                    df = mins_data[ticker][date]
                    if df.shape[0]>0:
                        df = df[df.Abnormal == 0]
                        df['group'] = np.floor(df.index // mins)
                        df['ret'] = df['Close'].diff(1)/df['Close'].shift(1)
                        df['valper'] = df['TradeValue']/df['TradeCount']
                        mins_df_group = df.groupby('group')

                        for calc in feature_names:
                            res = list(functions[calc](mins_df_group))
                            if len(res) ==240//mins:
                                data_df[calc].loc[ticker,date,:] = res
                except Exception as e:
                    print(e)
                    
        return data_df

