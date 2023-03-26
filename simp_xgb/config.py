import os
import numpy as np
import pandas as pd
'''
———————— 参数设置区 ————————
'''
kick_out_thrs = 0
co_use_factor_list = [
    'jxma_daily_pv_alpha53', 'jxma_intraday_pv_alpha16', 'yhzhou_alpha079',
    'yhzhou_alpha093', 'yhzhou_alpha062', 'yhzhou_alpha100', 'yhzhou_alpha086',
    'yhzhou_alpha082', 'jxma_daily_pv_alpha35', 'yhzhou_alpha020',
    'jxma_daily_pv_alpha76', 'yhzhou_alpha089', 'yhzhou_alpha074',
    'yhzhou_alpha092', 'yhzhou_alpha115', 'yhzhou_alpha076',
    'jxma_daily_pv_alpha31', 'yhzhou_alpha099', 'yhzhou_alpha083',
    'yhzhou_alpha091', 'jxma_daily_pv_alpha26', 'yhzhou_alpha102',
    'yhzhou_alpha103', 'yhzhou_alpha040', 'yhzhou_alpha107',
    'jxma_daily_pv_alpha46', 'yhzhou_alpha090', 'yhzhou_alpha088',
    'yhzhou_alpha067', 'yhzhou_alpha094', 'jxma_daily_pv_alpha20',
    'yhzhou_alpha113', 'yhzhou_alpha077', 'jxma_daily_pv_alpha17',
    'yhzhou_alpha073', 'jxma_daily_pv_alpha54', 'yhzhou_alpha060',
    'yhzhou_alpha064', 'yhzhou_alpha106', 'yhzhou_alpha112', 'yhzhou_alpha066',
    'yhzhou_alpha065', 'jxma_daily_pv_alpha11', 'jxma_intraday_pv_alpha4',
    'jxma_daily_pv_alpha69', 'jxma_daily_pv_alpha1', 'jxma_daily_pv_alpha55',
    'yhzhou_alpha063', 'yhzhou_alpha036', 'jxma_daily_pv_alpha18',
    'yhzhou_alpha114', 'yhzhou_alpha024', 'jxma_daily_pv_alpha5',
    'yhzhou_alpha070', 'jxma_daily_pv_alpha19', 'jxma_intraday_pv_alpha2',
    'yhzhou_alpha045', 'yhzhou_alpha105', 'jxma_intraday_pv_alpha12',
    'yhzhou_alpha075', 'jxma_intraday_pv_alpha9', 'yhzhou_alpha021',
    'jxma_daily_pv_alpha12', 'yhzhou_alpha104', 'yhzhou_alpha026',
    'jxma_daily_pv_alpha47', 'jxma_daily_pv_alpha16', 'yhzhou_alpha087'
]
shining_facs = [
    i[:-3] for i in os.listdir(
        '/data/shared/low_fre_alpha/yhzhou_shining_garden/eod_feature/')
]
yhzhou_num_list = [
    i for i in range(1, 116)
    if i not in [6, 7, 19, 20, 25, 28, 29, 30, 31, 36, 37, 40, 42, 44, 50]
]
yhzhou_factor_list = [
    f'eod_yhzhou_alpha{str(k).zfill(3)}' for k in yhzhou_num_list
]
tzni_factor_list = [
    'eod_' + i for i in [
        'miner1', 'miner10', 'miner101', 'miner102', 'miner103', 'miner104',
        'miner105', 'miner107', 'miner108', 'miner109', 'miner11', 'miner110',
        'miner111', 'miner112', 'miner113', 'miner114', 'miner115', 'miner116',
        'miner117', 'miner118', 'miner119', 'miner12', 'miner121', 'miner122',
        'miner123', 'miner124', 'miner125', 'miner126', 'miner127', 'miner128',
        'miner129', 'miner13', 'miner130', 'miner131', 'miner132', 'miner133',
        'miner134', 'miner136', 'miner137', 'miner138', 'miner139', 'miner14',
        'miner140', 'miner141', 'miner142', 'miner143', 'miner144', 'miner146',
        'miner147', 'miner149', 'miner15', 'miner150', 'miner151', 'miner152',
        'miner154', 'miner155', 'miner156', 'miner157', 'miner158', 'miner159',
        'miner16', 'miner160', 'miner161', 'miner162', 'miner163', 'miner164',
        'miner165', 'miner166', 'miner167', 'miner17', 'miner170', 'miner171',
        'miner172', 'miner173', 'miner174', 'miner175', 'miner176', 'miner177',
        'miner178', 'miner179', 'miner18', 'miner180', 'miner181', 'miner182',
        'miner185', 'miner187', 'miner188', 'miner189', 'miner190', 'miner191',
        'miner192', 'miner193', 'miner194', 'miner195', 'miner196', 'miner198',
        'miner2', 'miner20', 'miner200', 'miner201', 'miner202', 'miner203',
        'miner204', 'miner205', 'miner206', 'miner207', 'miner208', 'miner209',
        'miner21', 'miner210', 'miner211', 'miner212', 'miner213', 'miner214',
        'miner215', 'miner216', 'miner217', 'miner219', 'miner22', 'miner220',
        'miner221', 'miner222', 'miner223', 'miner224', 'miner225', 'miner226',
        'miner227', 'miner228', 'miner229', 'miner23', 'miner230', 'miner231',
        'miner232', 'miner237', 'miner238', 'miner24', 'miner240', 'miner241',
        'miner244', 'miner245', 'miner246', 'miner247', 'miner249', 'miner25',
        'miner250', 'miner251', 'miner252', 'miner253', 'miner255', 'miner256',
        'miner257', 'miner258', 'miner259', 'miner26', 'miner260', 'miner261',
        'miner262', 'miner264', 'miner265', 'miner266', 'miner267', 'miner268',
        'miner269', 'miner27', 'miner270', 'miner271', 'miner274', 'miner275',
        'miner276', 'miner278', 'miner28', 'miner280', 'miner281', 'miner282',
        'miner283', 'miner284', 'miner285', 'miner286', 'miner287', 'miner288',
        'miner289', 'miner29', 'miner290', 'miner291', 'miner292', 'miner293',
        'miner294', 'miner295', 'miner296', 'miner298', 'miner299', 'miner3',
        'miner30', 'miner300', 'miner301', 'miner302', 'miner303', 'miner304',
        'miner305', 'miner306', 'miner307', 'miner308', 'miner309', 'miner31',
        'miner310', 'miner311', 'miner312', 'miner313', 'miner314', 'miner316',
        'miner317', 'miner318', 'miner319', 'miner32', 'miner320', 'miner322',
        'miner323', 'miner324', 'miner325', 'miner326', 'miner327', 'miner328',
        'miner329', 'miner33', 'miner330', 'miner331', 'miner332', 'miner334',
        'miner335', 'miner336', 'miner338', 'miner339', 'miner34', 'miner340',
        'miner344', 'miner345', 'miner346', 'miner347', 'miner348', 'miner349',
        'miner35', 'miner350', 'miner351', 'miner352', 'miner353', 'miner355',
        'miner357', 'miner359', 'miner36', 'miner360', 'miner361', 'miner362',
        'miner363', 'miner365', 'miner366', 'miner367', 'miner369', 'miner37',
        'miner370', 'miner371', 'miner372', 'miner373', 'miner374', 'miner375',
        'miner376', 'miner377', 'miner378', 'miner38', 'miner380', 'miner39',
        'miner4', 'miner40', 'miner41', 'miner42', 'miner43', 'miner44',
        'miner45', 'miner46', 'miner47', 'miner48', 'miner49', 'miner5',
        'miner50', 'miner52', 'miner53', 'miner54', 'miner55', 'miner56',
        'miner57', 'miner58', 'miner59', 'miner6', 'miner61', 'miner62',
        'miner63', 'miner64', 'miner65', 'miner66', 'miner67', 'miner68',
        'miner69', 'miner7', 'miner70', 'miner71', 'miner73', 'miner75',
        'miner76', 'miner77', 'miner78', 'miner79', 'miner8', 'miner81',
        'miner82', 'miner83', 'miner85', 'miner86', 'miner88', 'miner9',
        'miner90', 'miner91', 'miner92', 'miner93', 'miner94', 'miner95',
        'miner98', 'miner99'
    ]
]
jxma_factor_list = [
    i[:-3]
    for i in os.listdir(
        '/data/shared/low_fre_alpha/00 factor_commit/eod_feature')
    if ('jxma' in i) and (('pv' in i) or ('combo' in i))
]
yhzhou_sp_list = [
    'eod_yhzhou_sp_alpha001', 'eod_yhzhou_sp_alpha002',
    'eod_yhzhou_sp_alpha003', 'eod_yhzhou_sp_alpha004'
]

# tzni list 106
list106 = ['eod_fac{}'.format(i) for i in range(1, 106)]
list108 = ['eod_fac{}'.format(i) for i in range(1, 108)]

# zz1800 list/factor_test_105/zz1800_bm500.csv', index_col=0)
zz1800_df = pd.read_csv(
    '/home/yhzhou/factor_test_105/fac_test_equal_400_bm_zz500_1800.csv',
    index_col=0)
zz1800_df['CEOscore'] = zz1800_df['AlphaSharpeNC'] * 0.01 - 0.2 * zz1800_df['AlphaDrawdownNC'] + \
    zz1800_df['AlphaRetNC'] * 0.1 - \
    zz1800_df['TurnOver'] / 100 + np.abs(zz1800_df['IC'])
zz1800_df['score'] = zz1800_df['AlphaSharpeNC'] * 0.005 - 0.5 * zz1800_df['AlphaDrawdownNC'] + \
    zz1800_df['AlphaRetNC'] * 1 - zz1800_df['TurnOver'] / \
    10 + np.abs(zz1800_df['IC'])
zz1800_list = [
    'eod_' + i for i in list(zz1800_df.iloc[:-4, :].sort_values(
        'AlphaSharpeNC', ascending=False).index)[:250]
]

# IC list
IC_df = pd.read_csv('/home/yhzhou/factor_test_105/IC_df_1228.csv', index_col=0)
# ic_list = ['eod_' + i for i in list(IC_df[(IC_df['rankic_lq_1'].abs() > 0.02) & (IC_df['rankic_lq_30'].abs() > 0.08)].index)]
ic_list = [
    'eod_' + i for i in list(IC_df.iloc[:-4, :].abs().sort_values(
        'rankic_lq_1', ascending=False).index)[:250]
]

# 1-day list
day1_list = [
    'eod_fac{}'.format(i) for i in [
        6, 14, 17, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 44, 45, 56,
        58, 61, 64, 81, 89
    ]
]
day1_path = '/data/shared/low_fre_alpha/factor_miner_results/1230_stats_tovr_gt30_to_yh_105'

# 模型小参数
reg_na_pct = 0.5
cls_na_pct = 0.5

# 因子方式
factor_method = 'raw'  # raw / linear / normal
'''
———————— 模型参数设置区 ————————
'''
# 模型评价函数


def xgb_feval_func(pred_arr, dmatrix):
    real_arr = dmatrix.get_label()
    idx = ~(np.isnan(real_arr) | np.isinf(real_arr) | np.isnan(pred_arr) |
            np.isinf(pred_arr))
    if len(pred_arr[idx]) > 0:
        ic = np.corrcoef(pred_arr[idx], real_arr[idx])[0, 1]
    else:
        ic = 0
    return 'eval-ic', -ic


# 分类参数
cls_xgb_paras = {
    'num_round': 150,
    'max_depth': 7,
    'eta': 0.06,
    'subsample': 1,  # 随机采样训练样本 训练实例的子采样比
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1-0.2
    'reg_lambda': 1,  # 控制模型复杂度的权重值的L2正则化项参数
    'max_delta_step': 0,
    'colsample_bytree': 1,  # 生成树时进行的列采样
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'objective': 'binary:logistic'
}
clsXgbParams = {
    'num_boost_round': cls_xgb_paras['num_round'],
    'verbose_eval': int(cls_xgb_paras['num_round'] / 3),
    'maximize': False,
    'params': cls_xgb_paras,
    'feval': xgb_feval_func
}
cls_xgb_paras.pop('num_round')
clean_std_small_range = 0.3
clean_std_large_range = 8

# 分类因子路径
cls_factor_path = '/data/shared/low_fre_alpha/yhzhou_shining_garden'
cls_factor_total_list = [
    i[:-3] for i in os.listdir(
        f'/data/shared/low_fre_alpha/yhzhou_shining_garden/eod_feature/')
]
# cls_factor_list = ['eod_yhzhou_alpha{}'.format(str(i).zfill(3)) for i in ([1, 3, 4, 5, 6] + list(np.arange(15, 28)) + list(np.arange(32, 116)))]
# cls_factor_list = cls_factor_list + [i for i in cls_factor_total_list if 'jxma' in i]
# cls_factor_list = ['eod_' + i for i in co_use_factor_list]
cls_factor_list = shining_facs
cls_switch = 'single'  # multi/single
cls_mode = 'expanding'  # expanding/rolling

# 回归参数
reg_xgb_paras = {
    'num_round': 150,
    'max_depth': 7,
    'eta': 0.06,
    'subsample': 1,  # 随机采样训练样本 训练实例的子采样比
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1-0.2
    'reg_lambda': 1,  # 0.1控制模型复杂度的权重值的L2正则化项参数
    'max_delta_step': 0,
    'colsample_bytree': 1,  # 生成树时进行的列采样
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'objective': 'reg:squarederror'
}
regXgbParams = {
    'num_boost_round': reg_xgb_paras['num_round'],
    'verbose_eval': int(reg_xgb_paras['num_round'] / 3),
    'maximize': False,
    'params': reg_xgb_paras,
    'feval': xgb_feval_func
}
reg_xgb_paras.pop('num_round')

# 回归因子路径
reg_factor_path = '/data/shared/low_fre_alpha/yhzhou_shining_garden'
# reg_factor_path = '/data/shared/low_fre_alpha/factor_miner_results/0104_test_zz500_0.8_q0.4_108'
reg_factor_total_list = [
    i[:-3] for i in os.listdir(
        f'/data/shared/low_fre_alpha/yhzhou_shining_garden/eod_feature/')
]
# reg_factor_list = ['eod_yhzhou_alpha{}'.format(str(i).zfill(3)) for i in ([1, 3, 4, 5, 6] + list(np.arange(15, 28)) + list(np.arange(32, 116)))]
# reg_factor_list = reg_factor_list + [i for i in reg_factor_total_list if 'jxma' in i]
# reg_factor_list = ['eod_' + i for i in co_use_factor_list]
reg_factor_list = reg_factor_total_list
reg_switch = 'single'  # multi/single
reg_mode = 'expanding'  # expanding/rolling

# Y值路径
return_len = 1
# memo = 'Ret {} with zz1800 bm500 RIC_1>2% RIC_30>8% on lq. Expanding.'.format(return_len)
# memo = 'Ret {} with zz1800 bm500 RIC_1 rank250 on lq. Expanding.'.format(return_len)
cls_memo = 'Ret {} with All Factors on lq. Factor {}. Expanding. Clean [{}, {}]*std Y.'.format(
    return_len, factor_method, clean_std_small_range, clean_std_large_range)
reg_memo = 'Ret {} with tzni new 108 on lq. Expanding.'.format(return_len)
return_name = 'eod_lq_80_60_rtn{}_opentwap_ex500'.format(return_len)
return_path = '/data/shared/low_fre_alpha/return_data'
