
import os
from functools import partial
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from PqiDataSdk import *


from generator import generator

ds = PqiDataSdk(size=1)

start_date = '20170101'
end_date = '20201231'
mins = 30
tickers = ds.get_ticker_list()
data_path = '/home/zywang/17 intraday/03 data_gen/feature_gen/'

def run(tickers):
    random.shuffle(tickers)
    ticker_batch = np.array_split(tickers,len(tickers)/2)
    with mp.Pool(processes = 200) as pool:
        result = list(tqdm(pool.imap(partial(gen.calc_mins,
                                             start_date=start_date,
                                             end_date=end_date), 
                                     list(ticker_batch)),
                            total=len(ticker_batch)))
    final_result = pd.concat(result).sort_index()
    print(final_result.head())
    final_result.to_parquet(data_path+'mins_30_features.parquet')
    



gen = generator()
run(tickers)