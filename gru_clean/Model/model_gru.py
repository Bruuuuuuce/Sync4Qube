if True:
    import gc
    import json
    import multiprocessing as mp
    import os
    import sys
    import time
    import warnings
    from functools import partial

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    warnings.filterwarnings("ignore")
    sys.path.append('/home/zywang/04 Combination/gru/mp_dev/gru_clean/')

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow_addons as tfa
    from Config.config import *
    from PqiData import *
    from PqiDataSdk import *
    from tensorflow import keras
    from Utils.gru_base import *
    from Utils.gru_multi import *
    from Utils.losses import *
    from Utils.rnn_attention import *
    from Utils.transformers import *

    ds = PqiDataSdk(size=1)

if config_dict['random_seed']:
    tf.random.set_seed(config_dict['random_seed'])
    np.random.seed(config_dict['random_seed'])

tickers_all_eod = ds.get_index_weight(ticker='000985',
                                      start_date=config_dict['start_date'],
                                      end_date=config_dict['end_date'],
                                      format='eod')
tickers_zz1000_eod = ds.get_index_weight(ticker='000852',
                                         start_date=config_dict['start_date'],
                                         end_date=config_dict['end_date'],
                                         format='eod')
tickers_zz500_eod = ds.get_index_weight(ticker='000905',
                                        start_date=config_dict['start_date'],
                                        end_date=config_dict['end_date'],
                                        format='eod')
tickers_zz800_eod = ds.get_index_weight(ticker='000906',
                                        start_date=config_dict['start_date'],
                                        end_date=config_dict['end_date'],
                                        format='eod')
tickers_hs300_eod = ds.get_index_weight(ticker='000300',
                                        start_date=config_dict['start_date'],
                                        end_date=config_dict['end_date'],
                                        format='eod')
tickers_zz1500_eod = (tickers_zz500_eod.replace(np.nan, 0) +
                      tickers_zz1000_eod.replace(np.nan, 0)).replace(0, np.nan)
tickers_zz1800_eod = (tickers_zz800_eod.replace(np.nan, 0) +
                      tickers_zz1000_eod.replace(np.nan, 0)).replace(0, np.nan)

dict_index = {
    'all': tickers_all_eod,
    '000852': tickers_zz1000_eod,
    '000905': tickers_zz500_eod,
    '000906': tickers_zz800_eod,
    '000300': tickers_hs300_eod,
    'zz1500': tickers_zz1500_eod,
    'zz1800': tickers_zz1800_eod
}

dict_model = {
    'gru_base': build_gru_base,
    'gru_multi': build_gru_multi,
    'gru_attn': build_gru_attn,
    'gru_selfattn': build_gru_selfattn,
    'transformer_base': build_transformer_base
}


def get_index_weight_eod(pool, start_date, end_date):
    df_index = dict_index[pool]
    df_index_select = df_index.loc[:, start_date:end_date].dropna(how='all')
    return df_index_select


def generate_train_dataset_ts(df_select,
                              lst_date,
                              valid=30,
                              n_past=20,
                              mins=30):
    idx = lst_date
    split = len(idx) - valid
    len_arr = 240 // mins * n_past
    index = df_select.index
    # arr = df_select.values
    X_train, y_train, X_valid, y_valid = [[] for _ in range(4)]

    i = n_past
    while i < split:
        arr_temp = df_select.loc[idx[i - n_past]:idx[i - 1]].values
        if arr_temp.shape[0] == len_arr:
            X_train.append(arr_temp[:, :-1])
            y_train.append(arr_temp[:, -1][-1])
        i += 1

    while i < len(lst_date):
        arr_temp = df_select.loc[idx[i - n_past]:idx[i - 1]].values
        if arr_temp.shape[0] == len_arr:
            X_valid.append(arr_temp[:, :-1])
            y_valid.append(arr_temp[:, -1][-1])
        i += 1

    return (np.array(X_train).astype('float32'),
            np.array(y_train).astype('float32'),
            np.array(X_valid).astype('float32'),
            np.array(y_valid).astype('float32'))


def generate_test_dataset_ts(input_data, lst_date, n_past=20, mins=30):
    ticker = input_data[0]
    df_select = input_data[1]
    index = df_select.index
    # arr = df_select.values
    idx = lst_date
    len_arr = 240 // mins * n_past

    X_test, y_test, y_test_idx = [[] for _ in range(3)]

    i = n_past
    while i < len(lst_date):
        arr_temp = df_select.loc[idx[i - n_past]:idx[i - 1]].values
        if arr_temp.shape[0] == len_arr:
            X_test.append(arr_temp[:, :-1])
            y_test.append(arr_temp[:, -1][-1])
            y_test_idx.append([ticker, idx[i - 1]])
        i += 1

    return (np.array(X_test).astype('float32'),
            np.array(y_test).astype('float32'), y_test_idx)


class ModelTS:

    def __init__(self) -> None:
        self.cfg = config_dict.copy()
        self.ds = ds

        self.loss = 'mse'
        self.optimizer = tfa.optimizers.AdamW(learning_rate=3e-4,
                                              weight_decay=1e-6)

        self.earlyStop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=self.cfg['ES_min_delta'],
            patience=self.cfg['ES_patience'],
            verbose=0,
            mode='auto',
            restore_best_weights=self.cfg['ES_restore'])
        self.earlyStop_train = keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.0001,
            patience=4,
            verbose=1,
            mode='min',
            restore_best_weights=False)
        self.LRreduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.5,
                                                          patience=5,
                                                          verbose=0,
                                                          mode='auto',
                                                          min_delta=0,
                                                          cooldown=0,
                                                          min_lr=3e-5)

    def prepare_data(self, start_date, end_date):
        if self.cfg['verbose'] > 1:
            t = time.time()
            print('preparing data')

        df_tickers_select = get_index_weight_eod('all', start_date, end_date)

        df_hfFeature = pd.read_parquet(self.cfg['feature_path'])
        df_hfTarget = pd.read_parquet(self.cfg['target_path'])

        if len(df_hfFeature) == len(df_hfTarget):
            df_hfFeature['ret'] = df_hfTarget.values
        else:
            df_hfFeature['ret'] = df_hfTarget

        df_hfAll = df_hfFeature

        df_hfAll = df_hfAll.loc[df_tickers_select.index.intersection(
            df_hfAll.index.levels[0])].loc[:, df_tickers_select.columns, :]

        frame = pd.DataFrame(0,
                             index=pd.MultiIndex.from_product([
                                 df_tickers_select.index,
                                 df_tickers_select.columns,
                                 range(8)
                             ]),
                             columns=df_hfAll.columns)
        self.df_hfAll_with_na = frame + df_hfAll
        # self.nd_hfAll_with_na = self.df_hfAll_with_na.reset_index().values

        self.df_hfAll = df_hfAll.dropna()
        self.df_hfAll.index = self.df_hfAll.index.remove_unused_levels(
        )  # 这是一个天坑
        self.idx_hfAll = self.df_hfAll.index
        # self.nd_hfAll = self.df_hfAll.reset_index().values
        if self.cfg['verbose'] > 1:
            print('Time Cost:{}s'.format(np.round(time.time() - t, 2)))

    def get_train_data(self, start_date, end_date):
        if self.cfg['verbose'] > 1:
            t = time.time()
            print('Train data')
        tickers_pool = get_index_weight_eod(self.cfg['pool_train'], start_date,
                                            end_date).index

        idx_lv0 = self.idx_hfAll.levels[0].intersection(tickers_pool)
        idx_lv1 = self.ds.get_trade_dates(start_date=start_date,
                                          end_date=end_date)
        result = []
        for ticker in idx_lv0:
            try:
                result.append(
                    generate_train_dataset_ts(self.df_hfAll.loc[ticker],
                                              lst_date=idx_lv1,
                                              valid=self.cfg['validation']))
            except:
                print(ticker)
        X_train = np.concatenate([res[0] for res in result if res[0].size != 0])
        y_train = np.concatenate([res[1] for res in result if res[1].size != 0])
        X_valid = np.concatenate([res[2] for res in result if res[2].size != 0])
        y_valid = np.concatenate([res[3] for res in result if res[3].size != 0])
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        if self.cfg['shuffle']:
            dataset_train = dataset_train.shuffle(buffer_size=int(1e6))
            dataset_valid = dataset_valid.shuffle(buffer_size=int(1e6))
        dataset_train = dataset_train.batch(self.cfg['batchsize'])
        dataset_valid = dataset_valid.batch(self.cfg['batchsize'])
        if self.cfg['verbose'] > 1:
            print('Time Cost:{}s'.format(np.round(time.time() - t, 2)))
        return dataset_train, dataset_valid, y_train, y_valid

    def get_test_data(self, start_date, end_date):
        if self.cfg['verbose'] > 1:
            t = time.time()
            print('Test data')
        # 限定ticker和date范围
        tickers_pool = get_index_weight_eod(self.cfg['pool_infer'], start_date,
                                            end_date).index
        for i in range(20):
            start_date = self.ds.get_prev_trade_date(trade_date=start_date)

        # idx_lv0 = tickers_pool
        idx_lv0 = self.idx_hfAll.levels[0].intersection(tickers_pool)
        idx_lv1 = self.ds.get_trade_dates(start_date=start_date,
                                          end_date=end_date)
        result = []
        for ticker in idx_lv0:
            result.append(
                generate_test_dataset_ts(
                    (ticker, self.df_hfAll_with_na.loc[ticker]),
                    lst_date=idx_lv1))
        X_test = np.concatenate([res[0] for res in result if res[0].size != 0])
        y_test = np.concatenate([res[1] for res in result if res[1].size != 0])
        y_test_idx = [res[2] for res in result if len(res[2]) != 0]

        dataset_test = tf.data.Dataset.from_tensor_slices(
            (X_test, y_test)).batch(self.cfg['batchsize'])
        if self.cfg['verbose'] > 1:
            print('Time Cost:{}s'.format(np.round(time.time() - t, 2)))
        return dataset_test, y_test, y_test_idx

    def build_model(self, **kwargs):
        return dict_model[self.cfg['model']](**kwargs)

    def get_df_pred(self, pred, real, idx):
        y_test_predict = pd.DataFrame()
        y_test_predict['y_pred'] = pd.Series(pred.reshape(-1))
        y_test_predict['y_real'] = pd.Series(real.reshape(-1))
        y_test_predict['ticker'] = [x[0] for x in idx]
        y_test_predict['date'] = [x[1] for x in idx]
        y_test_predict = y_test_predict.set_index(['ticker', 'date'])
        return y_test_predict

    def train_n_predict(self, dataset_train, dataset_valid, dataset_test,
                        y_train, y_test, y_test_index, train_start, train_end,
                        test_start, test_end):
        if self.cfg['verbose'] > 1:
            t = time.time()
            print('Train n Predict')
        idx = []
        for k in y_test_index:
            idx = idx + k
        y_pred_df = pd.DataFrame()
        y_pred_df_is = pd.DataFrame()  # 对训练集进行预测

        for i in range(self.cfg['repeat']):
            if self.cfg['verbose'] > 1:
                print(f"||Session {i+1}/{self.cfg['repeat']}||")

            model = self.build_model(input_shape=(160,
                                                  self.df_hfAll.shape[1] - 1),
                                     mlp_units=[64, 32, 8],
                                     unit_head=100,
                                     dropout=self.cfg['dropout'],
                                     mlp_dropout=self.cfg['mlp_dropout'])
            model.compile(loss=self.loss, optimizer=self.optimizer)

            history = model.fit(dataset_train,
                                validation_data=dataset_valid,
                                epochs=self.cfg['epochs'],
                                verbose=self.cfg['verbose'] - 1,
                                callbacks=[self.earlyStop, self.LRreduce])

            y_pred = model.predict(dataset_test)

            if self.cfg['save_model']:
                model.save(
                    save_path +
                    f"{self.cfg['model']}_{train_start}_{train_end}_{i}.h5")

            pred_df = pd.DataFrame()
            pred_df['y_pred'] = pd.Series(y_pred.reshape(-1))
            pred_df['ticker'] = [x[0] for x in idx]
            pred_df['date'] = [x[1] for x in idx]
            pred_df = pred_df.set_index(['ticker', 'date'])
            pred_df.index.names = [None, None]
            y_pred_df[i] = pred_df['y_pred']

        # y_pred_train = model.predict(dataset_train)[:, 0]
        # y_real_train = y_train
        # if self.cfg['verbose'] > 1:
        #     print('train corr',
        #           round(np.corrcoef(y_pred_train, y_real_train)[0, 1], 3))

        res_df = self.get_df_pred(
            model.predict(dataset_test)[:, 0], y_test, idx)
        if self.cfg['verbose'] > 1:
            print(
                'Test cs_IC:',
                round(
                    res_df['y_pred'].unstack().rank().corrwith(
                        res_df['y_real'].unstack()).mean(), 3))
            # print(model.evaluate(dataset_test))

        y_pred_res = y_pred_df.mean(axis=1).unstack()
        result = y_pred_res
        res_name = 'eod_gru_{}_{}_{}'.format(self.cfg['suffix'],
                                             result.columns[0],
                                             result.columns[-1])
        self.ds.save_eod_feature(data={res_name: result},
                                 where=save_path,
                                 feature_type='eod',
                                 encrypt=False,
                                 save_method='update')
        with open(save_path + f'config.json', 'w+') as f:
            json.dump(self.cfg, f, indent=4)
        if self.cfg['verbose'] > 0:
            print(res_name, 'saved.')
        if self.cfg['verbose'] > 1:
            print('Time Cost:{}s'.format(np.round(time.time() - t, 2)))

    def init_cuda(self, gpu_id):
        memory_limit = 6000
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[gpu_id], [
                tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=memory_limit)
            ])

    def run_one_model(self, train_start, train_end, test_start, test_end,
                      gpu_id):
        # try:
        self.init_cuda(gpu_id)
        self.prepare_data(min(train_start, test_start),
                          max(train_end, test_end))
        dataset_train, dataset_valid, y_train, y_valid = self.get_train_data(
            start_date=train_start, end_date=train_end)
        dataset_test, y_test, y_test_index = self.get_test_data(
            start_date=test_start, end_date=test_end)
        self.train_n_predict(dataset_train, dataset_valid, dataset_test,
                             y_train, y_test, y_test_index, train_start,
                             train_end, test_start, test_end)
        # except Exception as e:
        #     print('No prediction made for {}-{}'.format(test_start,test_end))
        #     print(e)


def run_single(train_start, train_end, test_start, test_end, gpu_id):
    if config_dict['verbose'] > 0:
        print('Train:{}-{}'.format(train_start, train_end),
              'Test:{}-{}'.format(test_start, test_end))
    model = ModelTS()
    model.run_one_model(train_start, train_end, test_start, test_end, gpu_id)


def run():
    # 主函数
    cfg = config_dict.copy()
    dates = ds.get_trade_dates(start_date=cfg['start_date'],
                               end_date=cfg['end_date'])
    train_start_idx = 0
    train_end_idx = train_start_idx + cfg['training']
    test_start_idx = train_end_idx + 3
    test_end_idx = test_start_idx + cfg['test'] - 1
    pool = mp.Pool(processes=cfg['gpu_num'])
    gpu_id = 0
    while test_end_idx < len(dates):
        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        pool.apply_async(run_single,
                         args=(train_start, train_end, test_start, test_end,
                               gpu_id))
        gpu_id += 1
        train_start_idx += cfg['test']
        train_end_idx += cfg['test']
        test_start_idx += cfg['test']
        test_end_idx += cfg['test']
        time.sleep(1)
        if gpu_id == cfg['gpu_num']:
            time.sleep(5)
            pool.close()
            pool.join()
            pool = mp.Pool(processes=cfg['gpu_num'])
            gpu_id = 0
    time.sleep(5)
    pool.close()
    pool.join()


def save_joined_factor(path, factor_save_path, factor_save_name, pool,
                       start_date, end_date, preffix, min_count):
    lst_trade_date = ds.get_trade_dates(start_date=start_date,
                                        end_date=end_date)
    lst_factor_name = sorted([
        name[:-3]
        for name in os.listdir(path + '/eod_feature')
        if name.startswith(preffix)
    ])

    data_factor = ds.get_eod_feature(fields=lst_factor_name,
                                     where=path,
                                     dates=lst_trade_date)
    df_factor = pd.DataFrame(index=data_factor.header[1],
                             columns=data_factor.header[2],
                             data=np.nansum(data_factor.values,
                                            axis=0)).replace(0, np.nan).dropna(
                                                how='all').dropna(1, how='all')
    df_idx_weight = get_index_weight_eod(pool,
                                         start_date=df_factor.columns[0],
                                         end_date=df_factor.columns[-1])
    df_idx_weight = df_idx_weight[df_idx_weight.index.isin(df_factor.index)]

    df_factor.loc[:, df_factor.count() < min_count] = df_idx_weight
    df_idx_weight.loc[:,
                      df_idx_weight.columns.isin(df_factor.columns)] = df_factor
    df_factor = df_idx_weight

    ds.save_eod_feature(data={factor_save_name: df_factor},
                        where=factor_save_path,
                        feature_type='eod',
                        save_method='update',
                        encrypt=False)