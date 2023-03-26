from register import Register

mins_functions = Register()

@mins_functions.register
def ret_mean(mins_df_group):
    return mins_df_group['ret'].mean()

@mins_functions.register
def ret_std(mins_df_group):
    return mins_df_group['ret'].std()

@mins_functions.register
def ret_skew(mins_df_group):
    return mins_df_group['ret'].skew()

@mins_functions.register
def tval_mean(mins_df_group):
    return mins_df_group['TradeValue'].mean()

@mins_functions.register
def tval_skew(mins_df_group):
    return mins_df_group['TradeValue'].skew()

@mins_functions.register
def count_mean(mins_df_group):
    return mins_df_group['TradeCount'].mean()

@mins_functions.register
def count_std(mins_df_group):
    return mins_df_group['TradeCount'].std()

@mins_functions.register
def count_skew(mins_df_group):
    return mins_df_group['TradeCount'].skew()

@mins_functions.register
def valper_mean(mins_df_group):
    return mins_df_group['valper'].mean()

@mins_functions.register
def valper_skew(mins_df_group):
    return mins_df_group['valper'].skew()

@mins_functions.register
def valper_skew(mins_df_group):
    return mins_df_group['valper'].skew()

@mins_functions.register
def net_act_buy(mins_df_group):
    act_buy = mins_df_group['ActBuyVol'].sum()
    act_sell = mins_df_group['ActSellVol'].sum()
    return (act_buy-act_sell)/(act_buy+act_sell)

@mins_functions.register
def act_buy_valper(mins_df_group):
    act_buy_val = mins_df_group['ActBuyVal'].sum()
    act_buy_cnt = mins_df_group['ActBuyCountGroupby'].sum()
    return act_buy_val/act_buy_cnt

@mins_functions.register
def act_sell_valper(mins_df_group):
    act_sell_val = mins_df_group['ActSellVal'].sum()
    act_sell_cnt = mins_df_group['ActSellCountGroupby'].sum()
    return act_sell_val/act_sell_cnt

trade_functions = Register()

@trade_functions.register
def actbuy_strenth(trade_df_group):
    pass






































