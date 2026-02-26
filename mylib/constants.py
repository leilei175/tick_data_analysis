"""
项目常量定义模块
存放所有高频使用的常量，避免重复定义
"""

from datetime import datetime

# =============================================================================
# 高频因子列表
# =============================================================================

HIGH_FREQUENCY_FACTORS = [
    'bid_ask_spread',       # 买卖价差
    'vwap_deviation',       # VWAP偏离度
    'trade_imbalance',      # 交易失衡度
    'order_imbalance',      # 订单失衡度
    'depth_imbalance',      # 深度失衡度
    'realized_volatility',  # 已实现波动率
    'effective_spread',     # 有效价差
    'micro_price',          # 微价格
    'price_momentum',       # 价格动量
    'trade_flow_intensity'  # 交易流强度
]

# 保持向后兼容
FACTORS = HIGH_FREQUENCY_FACTORS

# =============================================================================
# 季度日期列表 (2015-2026)
# =============================================================================

QUARTER_ENDS = [
    # 2015年
    '20150331', '20150630', '20150930', '20151231',
    # 2016年
    '20160331', '20160630', '20160930', '20161231',
    # 2017年
    '20170331', '20170630', '20170930', '20171231',
    # 2018年
    '20180331', '20180630', '20180930', '20181231',
    # 2019年
    '20190331', '20190630', '20190930', '20191231',
    # 2020年
    '20200331', '20200630', '20200930', '20201231',
    # 2021年
    '20210331', '20210630', '20210930', '20211231',
    # 2022年
    '20220331', '20220630', '20220930', '20221231',
    # 2023年
    '20230331', '20230630', '20230930', '20231231',
    # 2024年
    '20240331', '20240630', '20240930', '20241231',
    # 2025年
    '20250331', '20250630', '20250930', '20251231',
    # 2026年
    '20260331',
]

# =============================================================================
# 数据目录配置
# =============================================================================

DEFAULT_DATA_DIR = './daily_data/'
DEFAULT_FACTOR_DIR = './factor/'
DEFAULT_OUTPUT_DIR = './output/'

# 数据类型目录映射
DATA_TYPE_DIRS = {
    'daily': 'daily',
    'daily_basic': 'daily_basic',
    'cashflow': 'cashflow',
    'cashflow_daily': 'cashflow_daily',
    'income': 'income',
    'income_daily': 'income_daily',
    'balance': 'balance',
    'balance_daily': 'balance_daily',
}

# =============================================================================
# Tushare API字段配置
# =============================================================================

# 日线数据字段
DAILY_FIELDS = [
    'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
    'pre_close', 'change', 'pct_chg', 'vol', 'amount'
]

# 每日基本面数据字段
DAILY_BASIC_FIELDS = [
    'ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f',
    'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
    'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share',
    'total_mv', 'circ_mv'
]

# 利润表字段
INCOME_FIELDS = [
    'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
    'total_revenue', 'revenue', 'int_income', 'prem_earned', 'comm_income',
    'n_commis_income', 'operate_profit', 'total_profit', 'income_tax',
    'n_income', 'n_income_attr_p', 'minority_gain',
    'basic_eps', 'diluted_eps', 'ebit', 'ebitda'
]

# 资产负债表字段
BALANCE_FIELDS = [
    'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
    'total_assets', 'total_liab', 'total_hldr_eqy_exc_min_int',
    'total_cur_assets', 'total_nca', 'total_cur_liab', 'total_ncl',
    'cash_reser_cb', 'trad_asset', 'notes_receiv', 'accounts_receiv',
    'inventories', 'fix_assets', 'intan_assets',
    'st_borr', 'lt_borr', 'notes_payable', 'accounts_payable'
]

# 现金流量表字段
CASHFLOW_FIELDS = [
    'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
    'n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act',
    'c_fr_sale_sg', 'c_paid_goods_s', 'c_paid_to_for_empl',
    'c_recp_borrow', 'proc_issue_bonds'
]

# =============================================================================
# 可视化配置
# =============================================================================

MATPLOTLIB_CONFIG = {
    'font.sans-serif': ['DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.figsize': (12, 6),
    'figure.dpi': 100,
}

# =============================================================================
# 分析参数配置
# =============================================================================

# 默认IC计算参数
DEFAULT_IC_PARAMS = {
    'method': 'spearman',
    'min_samples': 10,
}

# 默认分层分析参数
DEFAULT_QUANTILE_PARAMS = {
    'n_quantiles': 5,
    'min_stocks': 100,
}

# 默认收益率计算周期
DEFAULT_RETURN_PERIODS = [1, 5, 10]

# =============================================================================
# 文件命名模式
# =============================================================================

# 因子文件命名模式
FACTOR_FILE_PATTERN = "zz1000_{factor}_{year}.parquet"
FACTOR_FILE_PATTERN_DAILY = "zz1000_factors_{date}.parquet"

# 数据文件命名模式
DAILY_FILE_PATTERN = "daily_{date}.parquet"
DAILY_BASIC_FILE_PATTERN = "daily_basic_{date}.parquet"

# =============================================================================
# 时间配置
# =============================================================================

# 默认日期范围
DEFAULT_START_DATE = '20250101'
DEFAULT_END_DATE = datetime.now().strftime('%Y%m%d')

# API请求间隔（秒）
API_REQUEST_INTERVAL = 0.1

# 交易日市场
TRADE_CAL_EXCHANGE = 'SSE'

# =============================================================================
# 错误处理和日志
# =============================================================================

# 是否过滤警告
FILTER_WARNINGS = True

# 默认日志级别
DEFAULT_LOG_LEVEL = 'INFO'

# 数据缺失值填充
DEFAULT_FILL_VALUE = 0.0
