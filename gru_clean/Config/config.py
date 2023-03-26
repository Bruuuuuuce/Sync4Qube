import os

config_dict = {
    'start_date':
        '20180101',
    'end_date':
        '20220731',
    'feature_path':
        '/home/shared/Data/data/shared/low_fre_alpha/zy_shared/gru/data/hfFeature_30min_v2_final_fill0_fp32.parquet',
    'target_path':
        '/home/shared/Data/data/shared/low_fre_alpha/zy_shared/gru/data/hfRet_rank_3d_30min.parquet',
    'pool_train':  # 'all', 000852', '000905', '000906', '000300', 'zz1500', 'zz1800'
        'all',
    'pool_infer':
        'all',
    'model':  # 'gru_base', 'gru_multi', 'gru_attn', 'gru_selfattn', 'transformer_base'
        'transformer_base',
    'random_seed':
        None,
    'training':
        120,
    'validation':
        2,
    'test':
        10,
    'shuffle':
        True,
    'batchsize':
        512,
    'epochs':
        8,
    'repeat':
        2,
    'ES_min_delta':
        1e-5,
    'ES_patience':
        100,
    'ES_restore':
        True,
    'dropout':
        0.0,
    'mlp_dropout':
        0.5,
    'save_model':
        True,
    'gpu_num':
        6,
    'save_path':
        '/home/zyding/gru_clean',
    'suffix':
        'transformer_temp_8e',
    'save_joined_factor':
        False,
    'factor_save_path':
        '/home/zyding/factor_zyding/eod_hf',
    'factor_save_name':
        'eod_gru_test',
    'verbose':
        2
}

save_path = config_dict['save_path'] + '/res_test_{}/'.format(
    config_dict['suffix'])

if not os.path.exists(save_path):
    os.makedirs(save_path)
