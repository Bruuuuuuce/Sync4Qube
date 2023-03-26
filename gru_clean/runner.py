import sys

sys.path.append('/home/zywang/04 Combination/gru/mp_dev/gru_clean/')
import time

from Model.model_gru import *
from Config.config import *

t = time.time()

run_single('20200325', '20200917', '20180601', '20220731', 1)
# run_single('20200325', '20200917', '20200325', '20200917', 0)
# run_single('20180102', '20180703', '20180101', '20220731', 1)
# run_single('20190125', '20190726', '20190731', '20190813', 1)
# run_single('20200609', '20201204', '20201209', '20201222', 1)
# run_single('20190823', '20200226', '20200302', '20200313', 1)

# run()

if config_dict['save_joined_factor']:
    save_joined_factor(path=save_path,
                       factor_save_path=config_dict['factor_save_path'],
                       factor_save_name=config_dict['factor_save_name'],
                       pool=config_dict['pool'],
                       start_date=config_dict['start_date'],
                       end_date=config_dict['end_date'],
                       preffix='eod_gru',
                       min_count=0)
    print(f"Joined factor '{config_dict['factor_save_name']}' saved.")

print('Time cost:', time.time() - t)

# from PqiDataSdk import PqiDataSdk

# ds = PqiDataSdk(user="zyding", size=1, pool_type="mt")
# lst_trade_date = ds.get_trade_dates(start_date='20180101', end_date='20210630')

# for i in range(0, len(lst_trade_date) - 120, 60):
#     train_start = lst_trade_date[i]
#     train_end = lst_trade_date[i + 120]

#     run_single(train_start, train_end, '20180101', '20220731', 4)

#     save_joined_factor(
#         path=save_path,
#         factor_save_path=
#         '/mnt/ceph/low_freq_team/low_fre_alpha/zy_shared/gru/ding/factor_zyding/eod_hf_test/eod_transformer_test_1800_csnorm3d_2e/',
#         factor_save_name=f'eod_transformer_test_{train_start}_{train_end}',
#         pool=config_dict['pool_infer'],
#         start_date='20180701',
#         end_date='20220731',
#         preffix='eod_gru',
#         min_count=400)

# print('Time cost:', time.time() - t)