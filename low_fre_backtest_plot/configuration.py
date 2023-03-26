
'''
输入 & 回测config
'''
# 测试类型
# 包括两类，'signal'为权重信号；而'combo'为组合因子类型
test_type = 'combo'

# --combo名称和路径（有需要填写，可DIY）
# combo_path = '/home/yhzhou/optsolver_ds_nsun'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_v2'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_v3'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_ex500'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_nofund'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_planA'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_planA_fee0001'
# combo_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_param_search'
combo_path = '/data/shared/low_fre_alpha/paper_trading_combo_v2'
# combo_path = '/data/nas/eod_data/eod_factor_trading/dervied_factors_05'
# combo_name = 'eod_xgb_20211228100736_reg'
# combo_name = 'eod_xgb_20211227133912_reg'
# combo_name = 'eod_test_fac'
combo_name = 'eod_xgb_20220307162205_reg_eod_zz1800_TwapOpen_rtn3_3_linear_test_lq8060'
# combo_name = 'eod_xgb_202202141428'
# combo_name = 'eod_factor_all_equal_zz1800'
# combo_name = 'eod_factor_best_equal_zz1800'
# combo_name = 'eod_factor_all_wgt_zz1800'
# combo_name = 'eod_factor_best_wgt_zz1800'
# combo_name = 'eod_yhzhou_alpha001'
# combo_name = 'eod_factor_rank_equal_yhzhou_v3_lq'
# combo_name = 'eod_xgb_20220307162221_reg_eod_zz1800_TwapOpen_rtn1_3_linear_test_lq8060'

# --signal名称和路径（有需要填写，可DIY）
signal_name = 'eod_opt_total_OS_V2_planA_0.5_0.5'
# signal_name = 'eod_reg_zz1800_1351020_nofund_roll03_add_pospl_50150200'
# signal_name = 'eod_opt_all_roll_30_0.3_0.3'
# signal_path = '/home/yhzhou/03_data/data/optimized_result'
# signal_path = '/home/yhzhou/03_data/platform_test'
signal_path = '/home/yhzhou/optsolver_ds_nsun'
# signal_path = '/data/shared/low_fre_alpha/yhzhou_comb_factors_v2'
# signal_path = '/data/shared/low_fre_alpha/paper_trading_combo'
# signal_path = '/data/shared/low_fre_alpha/yhzhou_opt_result'

# 总起讫日期，为重要数据的读取时间，包括信号/因子，以及ret_df等，以防有缺漏
total_start_date = '20150101'
total_end_date = '20220305'
# 回测起讫日期
# start_date = '20170101'
# end_date = '20201231'
# start_date = '20170630'
start_date = '20180101'
# end_date = '20210331'
end_date = '20210331'


# 回测因子及信号的参数，return_type有6中可选，TwapOpen30/60/120/240及Open和Close
# 部分参数在回测信号时不需要被使用
# method可选equal/factor,
## -- equal表示头组等权，需要设定下述head；
## -- factor表示因子值加权，不需要设定head；
method = 'equal'
head = 400
cost = 0.0013
group_num = 20
benchmark = 'index'
index = '000905'
# return_type = 'Close_to_Close'
# return_type = 'TwapOpen120_to_TwapOpen120'
return_type = 'TwapOpen60_to_TwapOpen60'
# return_type = 'TwapOpen240_to_TwapOpen240'


'''
风格分析config
'''
# benchmark指数
bm_index = index
# 排序方式
rank_method = 'exposure' # 可以按照暴露和归因排序，exposure / attribute
# 信号类型
signal_type = 'val' # 有两个选项，传入的是手数选择 vol，传入的是市值选择val
# 信号测试类型，低频长周期回测应该只使用‘multi'，此不需要修改
signal_test_type = 'multi' # 有两个选项，单日交易测试选择single，长时段回测测试选择multi
# 多日日期
multi_start_date = start_date
multi_end_date = end_date # 只在signal_test_type = 'multi'的情形下使用
# 单日日期
single_date = '20211130' # 只在signal_test_type = 'single'的情形下使用


'''
画图config
'''
# 结果储存路径
save_fig_path = './res/'

