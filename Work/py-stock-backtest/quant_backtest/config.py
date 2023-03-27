from typing import List

class BaseConfig:

    strategy: str = None

    start_date: str = None
    end_date: str = None
    buy_cost: float = 0.00085
    sell_cost: float = 0.00185

    price: str = 'close'  # settling price
    pool: str = 'all'     # asset pool

    # exact backtesting config
    init_amount: float = 1000000



class CrossSectionalConfig(BaseConfig):

    strategy = 'cross-sectional/alpha'

    holding_periods: List[int] = []
    holding_percentage: float = 0.1
    ngroups: int = 10