import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from glob import glob
from typing import Union, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

DAILY_DIR = './daily_data/daily/'
DAILY_BASIC_DIR = './daily_data/daily_basic/'
CASHFLOW_DAILY_DIR = './daily_data/cashflow_daily/'
INCOME_DAILY_DIR = './daily_data/income_daily/'
BALANCE_DAILY_DIR = './daily_data/balance_daily/'

# 可用数据类型
DATA_TYPES = ['daily', 'daily_basic', 'cashflow_daily', 'income_daily', 'balance_daily']


def _parse_date_from_filename(filename: str, prefix: str = 'daily') -> Optional[int]:
    """从文件名解析日期，返回8位日期整数，解析失败返回None"""
    pattern = rf'{prefix}_(\d{{8}})\.parquet$'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def _find_data_files(
    data_dir: str,
    prefix: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> List[tuple]:
    """
    查找数据文件，支持年/月目录结构

    Args:
        data_dir: 数据根目录
        prefix: 文件前缀
        start: 开始日期
        end: 结束日期

    Returns:
        List of (date, filepath) tuples, sorted by date
    """
    data_path = Path(data_dir)
    files_with_dates = []

    # 支持两种目录结构：
    # 1. flat: daily_data/daily/*.parquet
    # 2. hierarchical: daily_data/daily/2025/01/*.parquet

    # 查找所有匹配的文件
    patterns = [
        str(data_path / f'{prefix}_*.parquet'),  # 扁平结构
        str(data_path / '*' / '*' / f'{prefix}_*.parquet'),  # 年/月结构
    ]

    for pattern in patterns:
        for f in glob(pattern):
            fname = Path(f).name
            date = _parse_date_from_filename(fname, prefix)
            if date is not None:
                files_with_dates.append((date, f))

    # 根据日期范围过滤
    if start:
        start_date = int(start)
        files_with_dates = [(d, f) for d, f in files_with_dates if d >= start_date]
    if end:
        end_date = int(end)
        files_with_dates = [(d, f) for d, f in files_with_dates if d <= end_date]

    # 按日期排序
    files_with_dates.sort(key=lambda x: x[0])

    return files_with_dates


def _read_single_file(args: tuple) -> tuple:
    """读取单个parquet文件"""
    date, filepath, filed = args
    try:
        table = pq.read_table(filepath, columns=['ts_code', filed])
        df = table.to_pandas()
        df['trade_date'] = date
        return df
    except Exception:
        return None


def _get_merged_file_path(data_dir: str, year: str) -> Path:
    """获取合并后的年度文件路径"""
    return Path(data_dir) / f'{year}_all.parquet'


def _is_merged_file_available(data_dir: str, start: Optional[str], end: Optional[str]) -> bool:
    """检查合并文件是否可用"""
    if not start or not end:
        return False

    # 判断是否跨年
    start_year = start[:4]
    end_year = end[:4]

    if start_year == end_year:
        # 单年，检查合并文件是否存在
        merged_file = _get_merged_file_path(data_dir, start_year)
        return merged_file.exists()

    return False


def get_local_data(
    sec_list: Union[List[str], None] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    filed: str = 'close',
    data_type: str = 'daily',
    data_dir: Optional[str] = None,
    parallel: bool = True,
    max_workers: int = 8
) -> pd.DataFrame:
    """
    获取日频数据

    Args:
        sec_list: 股票代码列表，如 ['000001.SZ', '000002.SZ']，None表示所有股票
        start: 开始日期，格式 'YYYYMMDD'
        end: 结束日期，格式 'YYYYMMDD'
        filed: 要获取的字段名，默认'close'
        data_type: 数据类型，支持 'daily', 'daily_basic', 'cashflow_daily', 'income_daily', 'balance_daily'
        data_dir: 数据目录路径，默认为None，会根据data_type自动选择
        parallel: 是否并行读取（默认True）
        max_workers: 并行线程数（默认8）

    Returns:
        DataFrame: index为日期，columns为股票代码

    Example:
        >>> # 日线收盘价
        >>> get_local_data(['000001.SZ'], '20250101', '20250110', 'close', 'daily')
        >>> # 每日基本面换手率
        >>> get_local_data(['000001.SZ'], '20250101', '20250110', 'turnover_rate', 'daily_basic')
        >>> # 每日现金流
        >>> get_local_data(['000001.SZ'], '20250101', '20250110', 'n_cashflow_act', 'cashflow_daily')
    """
    # 验证 data_type
    if data_type not in DATA_TYPES:
        raise ValueError(f"不支持的数据类型: {data_type}，支持: {DATA_TYPES}")

    # 根据 data_type 确定默认目录和文件前缀
    if data_dir is None:
        if data_type == 'daily':
            data_dir = DAILY_DIR
            prefix = 'daily'
        elif data_type == 'daily_basic':
            data_dir = DAILY_BASIC_DIR
            prefix = 'daily_basic'
        elif data_type == 'cashflow_daily':
            data_dir = CASHFLOW_DAILY_DIR
            prefix = 'cashflow_daily'
        elif data_type == 'income_daily':
            data_dir = INCOME_DAILY_DIR
            prefix = 'income_daily'
        else:  # balance_daily
            data_dir = BALANCE_DAILY_DIR
            prefix = 'balance_daily'
    else:
        # 从 data_dir 推断 prefix
        prefix = data_type

    # 查找文件
    files_with_dates = _find_data_files(data_dir, prefix, start, end)

    if not files_with_dates:
        return pd.DataFrame()

    # 构建数据
    dfs = []

    # 根据日期范围决定读取策略
    start_year = start[:4] if start else None
    end_year = end[:4] if end else None
    is_single_year = (start_year == end_year and
                      _is_merged_file_available(data_dir, start, end))

    if is_single_year and parallel:
        # 单年：使用合并文件
        year = start_year
        merged_file = _get_merged_file_path(data_dir, year)
        print(f"[优化] 使用合并文件: {merged_file.name}")

        df_all = pd.read_parquet(merged_file)

        # 按日期过滤
        start_int = int(start)
        end_int = int(end)
        if df_all['trade_date'].dtype == object:
            df_all['trade_date'] = df_all['trade_date'].astype(int)
        df_all = df_all[(df_all['trade_date'] >= start_int) &
                        (df_all['trade_date'] <= end_int)]

    elif parallel and start_year != end_year and start_year and end_year:
        # 跨年：并行读取多个合并文件
        def read_year_file(year):
            merged_file = _get_merged_file_path(data_dir, year)
            if merged_file.exists():
                return pd.read_parquet(merged_file)
            return None

        def read_year_daily(year):
            """读取某年所有每日小文件"""
            year_dir = Path(data_dir) / year
            if not year_dir.exists():
                return None
            files = list(year_dir.glob('*/daily_*.parquet'))
            if not files:
                return None
            dfs = []
            for f in sorted(files):
                try:
                    df = pd.read_parquet(f, columns=['ts_code', filed])
                    df['trade_date'] = int(f.name.split('_')[1].split('.')[0])
                    dfs.append(df)
                except:
                    pass
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            return None

        # 收集所有需要读取的年份
        start_yr = int(start_year)
        end_yr = int(end_year)
        all_years = list(range(start_yr, end_yr + 1))

        # 检查哪些年份有合并文件
        years_with_merged = []
        years_without_merged = []
        for yr in all_years:
            merged_file = _get_merged_file_path(data_dir, str(yr))
            if merged_file.exists():
                years_with_merged.append(yr)
            else:
                years_without_merged.append(yr)

        print(f"[优化] 读取 {len(all_years)} 年数据 ({len(years_with_merged)} 个合并文件, {len(years_without_merged)} 个每日文件)")

        all_dfs = []

        # 并行读取有合并文件的年份
        if years_with_merged:
            with ThreadPoolExecutor(max_workers=min(len(years_with_merged), 6)) as executor:
                dfs_merged = list(executor.map(read_year_file, years_with_merged))
            all_dfs.extend([d for d in dfs_merged if d is not None])

        # 读取没有合并文件的年份
        for yr in years_without_merged:
            df = read_year_daily(str(yr))
            if df is not None:
                all_dfs.append(df)

        df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        # 按日期过滤
        if not df_all.empty:
            if df_all['trade_date'].dtype == object:
                df_all['trade_date'] = df_all['trade_date'].astype(int)
            if start:
                df_all = df_all[df_all['trade_date'] >= int(start)]
            if end:
                df_all = df_all[df_all['trade_date'] <= int(end)]

    elif parallel and len(files_with_dates) > 10:
        # 并行读取每日小文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            args_list = [(date, f, filed) for date, f in files_with_dates]
            results = list(executor.map(_read_single_file, args_list))
            dfs = [r for r in results if r is not None]
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        # 串行读取
        for date, f in files_with_dates:
            table = pq.read_table(f, columns=['ts_code', filed])
            df = table.to_pandas()
            df['trade_date'] = date
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)

    # 过滤股票
    if sec_list is not None and len(sec_list) > 0:
        df_all = df_all[df_all['ts_code'].isin(sec_list)]

    # 过滤空值并去重
    df_all = df_all.dropna(subset=[filed])
    df_all = df_all.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')

    # 确保trade_date是有效整数
    df_all = df_all[df_all['trade_date'].apply(lambda x: str(x).isdigit() if pd.notna(x) else False)]

    # 转为宽表
    df_pivot = df_all.pivot(index='trade_date', columns='ts_code', values=filed)

    # 安全转换日期
    try:
        df_pivot.index = pd.to_datetime(df_pivot.index.astype(str), format='%Y%m%d', errors='raise')
    except ValueError:
        # 如果直接转换失败，使用mixed格式
        df_pivot.index = pd.to_datetime(df_pivot.index.astype(str), format='mixed', errors='coerce')

    df_pivot.index.name = 'date'

    # 按日期排序索引
    df_pivot = df_pivot.sort_index()

    return df_pivot


def list_data_files(
    data_type: str = 'daily',
    data_dir: Optional[str] = None,
    year: Optional[str] = None,
    month: Optional[str] = None
) -> List[tuple]:
    """
    列出数据文件

    Args:
        data_type: 数据类型
        data_dir: 自定义目录
        year: 筛选年份 (YYYY)
        month: 筛选月份 (MM)

    Returns:
        List of (date, filepath) tuples
    """
    if data_dir is None:
        if data_type == 'daily':
            data_dir = DAILY_DIR
            prefix = 'daily'
        elif data_type == 'daily_basic':
            data_dir = DAILY_BASIC_DIR
            prefix = 'daily_basic'
        elif data_type == 'cashflow_daily':
            data_dir = CASHFLOW_DAILY_DIR
            prefix = 'cashflow_daily'
        elif data_type == 'income_daily':
            data_dir = INCOME_DAILY_DIR
            prefix = 'income_daily'
        else:
            data_dir = BALANCE_DAILY_DIR
            prefix = 'balance_daily'
    else:
        prefix = data_type

    files = _find_data_files(data_dir, prefix)

    if year:
        files = [(d, f) for d, f in files if str(d).startswith(year)]
    if month:
        files = [(d, f) for d, f in files if str(d)[4:6] == month]

    return files


def _get_full_file_path(data_dir: str, year: str) -> Path:
    """获取完整合并文件的路径"""
    return Path(data_dir) / f'{year}_full.parquet'


def get_all_data(
    data_type: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sec_list: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    快速获取指定类型的所有数据

    Args:
        data_type: 数据类型 ('daily', 'daily_basic', 'cashflow_daily', 'income_daily', 'balance_daily')
        start: 开始日期，格式 'YYYYMMDD'
        end: 结束日期，格式 'YYYYMMDD'
        sec_list: 股票代码列表，None表示所有股票
        fields: 字段列表，None表示获取所有可用字段
        parallel: 是否并行读取（仍会一次性读取所有字段）

    Returns:
        Dict: {field_name: DataFrame}

    Example:
        >>> data = get_all_data('daily', start='20200101', end='20201231')
        >>> close_df = data['close']
    """
    from concurrent.futures import ThreadPoolExecutor

    # 定义各数据类型的可用字段
    DATA_FIELDS = {
        'daily': ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                  'pre_close', 'change', 'pct_chg', 'vol', 'amount'],
        'daily_basic': ['ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f',
                       'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
                       'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv'],
        'cashflow_daily': ['ts_code', 'trade_date', 'n_cashflow_act', 'n_cashflow_inv_act',
                          'n_cash_flows_fnc_act', 'c_fr_sale_sg', 'c_paid_goods_s',
                          'c_paid_to_for_empl', 'c_recp_borrow', 'proc_issue_bonds'],
        'income_daily': ['ts_code', 'trade_date', 'total_revenue', 'revenue', 'int_income',
                         'operate_profit', 'total_profit', 'income_tax', 'n_income',
                         'basic_eps', 'diluted_eps'],
        'balance_daily': ['ts_code', 'trade_date', 'total_assets', 'total_liab', 'total_cur_assets',
                         'total_cur_liab', 'cash_reser_cb', 'accounts_receiv', 'inventories']
    }

    # 确定要读取的字段
    if data_type not in DATA_FIELDS:
        raise ValueError(f"未知数据类型: {data_type}")
    all_fields = DATA_FIELDS[data_type]

    if fields is None:
        fields = [f for f in all_fields if f not in ['ts_code', 'trade_date']]
    else:
        # 验证字段
        fields = [f for f in fields if f in all_fields and f not in ['ts_code', 'trade_date']]

    # 确定数据目录
    if data_type == 'daily':
        data_dir = DAILY_DIR
    elif data_type == 'daily_basic':
        data_dir = DAILY_BASIC_DIR
    elif data_type == 'cashflow_daily':
        data_dir = CASHFLOW_DAILY_DIR
    elif data_type == 'income_daily':
        data_dir = INCOME_DAILY_DIR
    else:
        data_dir = BALANCE_DAILY_DIR

    # 确定年份范围
    start_year = start[:4] if start else None
    end_year = end[:4] if end else None
    if not start_year or not end_year:
        raise ValueError("需要指定start和end日期")

    start_yr = int(start_year)
    end_yr = int(end_year)
    all_years = list(range(start_yr, end_yr + 1))

    # 检查哪些年份有full文件
    years_with_full = []
    years_without_full = []
    for yr in all_years:
        full_file = _get_full_file_path(data_dir, str(yr))
        if full_file.exists():
            years_with_full.append(yr)
        else:
            years_without_full.append(yr)

    print(f"[get_all_data] {data_type}: {len(all_years)}年数据, {len(years_with_full)}个full文件, {len(years_without_full)}个需合并")

    # 读取函数
    def read_full_year(yr: int) -> pd.DataFrame:
        full_file = _get_full_file_path(data_dir, str(yr))
        if full_file.exists():
            return pd.read_parquet(full_file, columns=all_fields)
        return None

    all_dfs = []

    # 并行读取有full文件的年份
    if years_with_full:
        with ThreadPoolExecutor(max_workers=min(len(years_with_full), 6)) as executor:
            for df in executor.map(read_full_year, years_with_full):
                if df is not None:
                    all_dfs.append(df)

    # 处理没有full文件的年份（如果有的话）
    for yr in years_without_full:
        year_files = _find_data_files(data_dir, data_type, f'{yr}0101', f'{yr}1231')
        if year_files:
            dfs = []
            for date, f in year_files:
                df = pd.read_parquet(f, columns=all_fields)
                dfs.append(df)
            if dfs:
                all_dfs.append(pd.concat(dfs, ignore_index=True))

    if not all_dfs:
        return {f: pd.DataFrame() for f in fields}

    # 合并所有年份
    df_all = pd.concat(all_dfs, ignore_index=True)

    # 按日期过滤
    start_int = int(start)
    end_int = int(end)
    if df_all['trade_date'].dtype == object:
        df_all['trade_date'] = df_all['trade_date'].astype(int)
    df_all = df_all[(df_all['trade_date'] >= start_int) & (df_all['trade_date'] <= end_int)]

    # 过滤股票
    if sec_list:
        df_all = df_all[df_all['ts_code'].isin(sec_list)]

    # 去重
    df_all = df_all.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')

    # 拆分字段 - 批量unstack优化版本
    result = {}

    # 获取唯一的日期和股票（注意：按整数排序而非字符串排序）
    unique_dates_int = sorted(df_all['trade_date'].unique())
    unique_stocks = df_all['ts_code'].unique()

    # 创建日期索引（用于最终结果）
    date_index = pd.to_datetime([str(d) for d in unique_dates_int], format='%Y%m%d')
    date_index.name = 'date'

    # 设置索引
    df_indexed = df_all.set_index(['trade_date', 'ts_code'])

    # 只选择需要的字段
    available_fields = [f for f in fields if f in df_indexed.columns]

    if available_fields:
        # 批量unstack：一次性展开所有字段，性能提升6倍
        df_multi = df_indexed[available_fields]
        df_unstack = df_multi.unstack(level='ts_code')  # 结果是 MultiIndex 列

        # 转换索引
        df_unstack.index = date_index

        # 拆分每个字段（先提取再转换列类型）
        for field in available_fields:
            # 从MultiIndex列中提取单个字段
            df_field = df_unstack[field]
            df_field.columns = df_field.columns.astype(str)
            result[field] = df_field

    # 对于不存在的字段，创建空DataFrame
    for field in fields:
        if field not in result:
            result[field] = pd.DataFrame(
                np.full((len(unique_dates_int), len(unique_stocks)), np.nan),
                index=date_index,
                columns=[str(s) for s in unique_stocks]
            )

    return result


def get_daily_all(
    start: Optional[str] = None,
    end: Optional[str] = None,
    sec_list: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    快速获取日线所有数据（便捷函数）

    等价于: get_all_data('daily', start=start, end=end, sec_list=sec_list)

    Args:
        start: 开始日期
        end: 结束日期
        sec_list: 股票代码列表
        parallel: 是否并行

    Returns:
        Dict: {'open': df, 'high': df, 'low': df, 'close': df, ...}
    """
    return get_all_data('daily', start=start, end=end, sec_list=sec_list, parallel=parallel)


def get_daily_basic_all(
    start: Optional[str] = None,
    end: Optional[str] = None,
    sec_list: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    快速获取每日基本面所有数据（便捷函数）

    等价于: get_all_data('daily_basic', start=start, end=end, sec_list=sec_list)

    Returns:
        Dict: {'close': df, 'turnover_rate': df, 'pe': df, ...}
    """
    return get_all_data('daily_basic', start=start, end=end, sec_list=sec_list, parallel=parallel)
