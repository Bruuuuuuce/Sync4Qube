from numba import jit
import numpy as np
import numba as nb


@nb.extending.overload(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if isinstance(x, nb.types.Array):
        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if copy:
                out = np.copy(x).reshape(-1)
            else:
                out = x.reshape(-1)
            for i in range(len(out)):
                if np.isnan(out[i]):
                    out[i] = nan
                if posinf is not None and np.isinf(out[i]) and out[i] > 0:
                    out[i] = posinf
                if neginf is not None and np.isinf(out[i]) and out[i] < 0:
                    out[i] = neginf
            return out.reshape(x.shape)

    else:
        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if np.isnan(x):
                return nan
            if posinf is not None and np.isinf(x) and x > 0:
                return posinf
            if neginf is not None and np.isinf(x) and x < 0:
                return neginf
            return x

    return nan_to_num_impl


@jit(nopython=True)
def run_strategy_cross_sectional_numba(alpha: np.ndarray, today_return: np.ndarray, 
    can_buy: np.ndarray, can_sell: np.ndarray, n_trading_days: int,
    pool_size: int, holding_period: int, buy_fee_ratio: float, 
    sell_fee_ratio: float, holding_percentage: float):
    """
    Run simple strategy backtests. alpha series need to be shifted.

    Consider:
    - transaction fee (assume deal at close price, and do not need to adjust portfolio due to transaction fee)
    - buy and sell limit

    Do not consider:
    - division rounding problem
    """
    portfolio_weights = np.zeros((n_trading_days, holding_period, pool_size))
    # the portfolio value is calculated after all transaction of the day (i.e., after the market is closed)
    raw_values = np.ones(n_trading_days) # before fee
    net_values = np.ones(n_trading_days) # after fee
    change_over = np.zeros(n_trading_days) # 单边换手
    full_position_status = np.zeros(n_trading_days)
    # the main loop
    for i in range(n_trading_days):
        # update portfolio weight
        group_id = i % holding_period
        # import pdb; pdb.set_trace()
        alp = alpha[i]
        alp_argindex = alp.argsort()
        alp_argindex = alp_argindex[~np.isnan(alp[alp_argindex])]
        alp_select = alp_argindex[-int(holding_percentage*len(alp_argindex)):]
        # consider buy and sell limits
        # consider sell limits
        if i > 0:
            # need to sell some portion
            cannot_sell = 1-np.nan_to_num(can_sell[i], 0)   # 可能出现停牌等导致中间数据确实，不能卖出
            cannot_sell *= portfolio_weights[i-1, group_id]
        else:
            cannot_sell = np.zeros_like(portfolio_weights[i, group_id])
        # import pdb; pdb.set_trace()
        cannot_sell_portion = cannot_sell.sum()
        for instr_index in alp_select:
            if cannot_sell[instr_index] == 0:   # 如果上个周期持仓，且这个周期不能卖出，仓位自动保留，在后面处理
                portfolio_weights[i, group_id, instr_index] = 1
        portfolio_weights[i, group_id] *= np.nan_to_num(can_buy[i], 0)
        full_position = True
        if portfolio_weights[i, group_id].sum() == 0:
            # 当天信号全为空，目前处理方法是保持上周期持仓
            # warnings.warn(f'Signal of day {i} are empty!')
            print(f'WARNING - Signal of day {i} are empty!')
            portfolio_weights[i, group_id] = portfolio_weights[i-1, group_id] \
                if i > 0 else np.zeros_like(portfolio_weights[i, group_id])
            full_position = False if portfolio_weights[i, group_id].sum() == 0 else True
        else:
            portfolio_weights[i, group_id] *= (1/holding_period - cannot_sell_portion) \
                / portfolio_weights[i, group_id].sum()
        portfolio_weights[i, group_id] += cannot_sell
        # import pdb; pdb.set_trace()
        # check for full position
        assert np.abs(portfolio_weights[i, group_id].sum() - 1/holding_period) < 1e-10 or (not full_position)
        # calculate transaction fee
        weight_diff = portfolio_weights[i, group_id] - portfolio_weights[i-1, group_id] \
            if i > 0 else portfolio_weights[i, group_id]
        change_over[i] = np.abs(weight_diff).sum() / 2
        buy_portion = weight_diff[weight_diff > 0].sum()
        sell_portion = -weight_diff[weight_diff < 0].sum()
        # update stock return to portfolio
        # copy other group weights
        if i > 0:
            for j in range(holding_period):
                if j != group_id:
                    portfolio_weights[i, j] = portfolio_weights[i-1, j]
        yesterday_weight = portfolio_weights[i-1].sum(axis=0) if i > 0 else np.zeros(pool_size)
        # assert np.abs(yesterday_weight.sum() - min(1, i/holding_period)) < 1e-10
        assert np.logical_or(weight_diff == 0, ~np.isnan(today_return[i])).all()
        today_ret = np.nansum(yesterday_weight * today_return[i])
        raw_values[i] = (raw_values[i-1] if i > 0 else 1) * (1 + today_ret)
        net_values[i] = (net_values[i-1] if i > 0 else 1) * (1 + today_ret)
        # reduct fee
        buy_fee = net_values[i] * buy_portion * buy_fee_ratio
        sell_fee = net_values[i] * sell_portion * sell_fee_ratio
        net_values[i] -= (buy_fee + sell_fee)

        full_position_status[i] = full_position
    # record results
    portfolio_weights = portfolio_weights.sum(axis=1)
    assert np.allclose(portfolio_weights.sum(axis=-1), 
                       np.minimum(1, np.cumsum(full_position_status) / holding_period), 
                       1e-10, 1e-12, False)
    # avg_value = (np.nanmean(today_return, axis=1)+1).cumprod()
    # today_return = np.ascontiguousarray(today_return)
    avg_value = np.empty_like(today_return)
    for j in range(pool_size):
        avg_value[:, j] = np.cumprod(np.nan_to_num(np.ascontiguousarray(today_return[:, j])) + 1)
    benchmark = np.zeros(n_trading_days)
    for j in range(n_trading_days):
        benchmark[j] = np.mean(avg_value[j, :])
    return net_values, raw_values, benchmark, change_over, portfolio_weights