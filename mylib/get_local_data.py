import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq
from glob import glob
from typing import Union, List, Optional

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


def get_local_data(
    sec_list: Union[List[str], None] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    filed: str = 'close',
    data_type: str = 'daily',
    data_dir: Optional[str] = None
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

    # 转为宽表
    df_pivot = df_all.pivot(index='trade_date', columns='ts_code', values=filed)
    df_pivot.index = pd.to_datetime(df_pivot.index, format='%Y%m%d')
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
