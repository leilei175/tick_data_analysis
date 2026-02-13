"""
Tushare 数据下载脚本
=================

功能：
- 从 Tushare 下载日线行情数据和每日基本面数据
- 保存为 Parquet 格式，与现有数据结构兼容

使用说明：
---------

1. 配置 Tushare Token（任选一种方式）：
   - 环境变量（推荐）: export TUSHARE_TOKEN='your_token'
   - 配置文件: ~/.tushare_token
   - 代码中设置: set_token('your_token')

2. 运行下载：

   # 下载日线数据
   python tushare_downloader.py daily --start 20260101 --end 20261231

   # 下载每日基本面数据
   python tushare_downloader.py daily_basic --start 20260101 --end 20261231

   # 下载所有数据
   python tushare_downloader.py all --start 20260101 --end 20261231

3. Python API 使用：

   from tushare_downloader import download_daily, download_daily_basic

   # 下载日线数据
   download_daily(
       start_date='20260101',
       end_date='20261231',
       output_dir='./daily_data/daily'
   )

   # 下载基本面数据
   download_daily_basic(
       start_date='20260101',
       end_date='20261231',
       output_dir='./daily_data/daily_basic'
   )
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Tushare
try:
    import tushare as ts
    HAS_TUSHARE = True
except ImportError:
    HAS_TUSHARE = False
    print("警告: 未安装 tushare，请运行: pip install tushare")

# =============================================================================
# 配置
# =============================================================================

# 默认数据保存目录
DEFAULT_DAILY_DIR = './daily_data/daily/'
DEFAULT_DAILY_BASIC_DIR = './daily_data/daily_basic/'

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

# =============================================================================
# Token 管理
# =============================================================================

def set_token(token: str):
    """
    设置 Tushare Token

    Args:
        token: Tushare API Token
    """
    ts.set_token(token)
    print(f"Token 已设置")


def get_token() -> str:
    """
    获取 Tushare Token

    优先级：
    1. 环境变量 TUSHARE_TOKEN
    2. ~/.tushare_token 文件
    3. 抛出异常

    Returns:
        str: Token 字符串
    """
    # 1. 环境变量
    token = os.environ.get('TUSHARE_TOKEN')
    if token:
        return token

    # 2. 配置文件
    token_path = Path.home() / '.tushare_token'
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    raise ValueError(
        "未找到 Tushare Token，请通过以下方式之一设置：\n"
        "1. 环境变量: export TUSHARE_TOKEN='your_token'\n"
        "2. 配置文件: echo 'your_token' > ~/.tushare_token\n"
        "3. 代码调用: set_token('your_token')"
    )


def init_tushare():
    """
    初始化 Tushare
    """
    if not HAS_TUSHARE:
        raise ImportError("未安装 tushare，请运行: pip install tushare")

    token = get_token()
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


# =============================================================================
# 数据下载函数
# =============================================================================

def download_daily(
    start_date: str,
    end_date: str,
    output_dir: str = DEFAULT_DAILY_DIR,
    fields: list = None,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    下载日线行情数据

    Args:
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        output_dir: 输出目录
        fields: 字段列表，None 表示使用默认字段
        max_workers: 并行下载线程数

    Returns:
        pd.DataFrame: 下载的数据

    Example:
        >>> download_daily('20260101', '20261231', './daily_data/daily')
    """
    if fields is None:
        fields = DAILY_FIELDS

    print(f"开始下载日线数据: {start_date} ~ {end_date}")
    pro = init_tushare()

    # 生成交易日列表
    trade_cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )
    trade_dates = trade_cal['cal_date'].tolist()
    print(f"交易日数量: {len(trade_dates)}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 按月分组，减少API调用
    def get_month_dates(dates):
        """将日期按月分组"""
        months = {}
        for d in dates:
            month = d[:6]
            if month not in months:
                months[month] = []
            months[month].append(d)
        return months

    month_groups = get_month_dates(trade_dates)
    print(f"按 {len(month_groups)} 个月份分组下载")

    all_data = []
    months_done = 0

    for month, dates in sorted(month_groups.items()):
        # 获取月份第一天和最后一天
        month_start = dates[0]
        month_end = dates[-1]

        # 下载当月数据
        df = pro.daily(
            start_date=month_start,
            end_date=month_end,
            fields=','.join(fields)
        )

        if df.empty:
            print(f"  {month}: 无数据")
            continue

        # 按日期保存为 Parquet
        for trade_dt in dates:
            day_df = df[df['trade_date'] == trade_dt]
            if day_df.empty:
                continue

            # 格式化日期用于文件名
            date_str = trade_dt
            filename = f"daily_{date_str}.parquet"
            filepath = output_path / filename

            # 保存为 Parquet
            table = pa.Table.from_pandas(day_df, preserve_index=False)
            pq.write_table(table, str(filepath))

        all_data.append(df)
        months_done += 1
        print(f"  {month}: {len(df)} 条记录")

        # 避免请求过快
        time.sleep(0.3)

    # 合并所有数据
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n下载完成! 总计 {len(result)} 条记录")
        return result
    else:
        print("\n未下载到任何数据")
        return pd.DataFrame()


def download_daily_basic(
    start_date: str,
    end_date: str,
    output_dir: str = DEFAULT_DAILY_BASIC_DIR,
    fields: list = None,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    下载每日基本面数据

    Args:
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        output_dir: 输出目录
        fields: 字段列表，None 表示使用默认字段
        max_workers: 并行下载线程数

    Returns:
        pd.DataFrame: 下载的数据

    Example:
        >>> download_daily_basic('20260101', '20261231', './daily_data/daily_basic')
    """
    if fields is None:
        fields = DAILY_BASIC_FIELDS

    print(f"开始下载每日基本面数据: {start_date} ~ {end_date}")
    pro = init_tushare()

    # 获取交易日列表
    trade_cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )
    trade_dates = trade_cal['cal_date'].tolist()
    print(f"交易日数量: {len(trade_dates)}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def get_month_dates(dates):
        """将日期按月分组"""
        months = {}
        for d in dates:
            month = d[:6]
            if month not in months:
                months[month] = []
            months[month].append(d)
        return months

    month_groups = get_month_dates(trade_dates)
    print(f"按 {len(month_groups)} 个月份分组下载")

    all_data = []

    for month, dates in sorted(month_groups.items()):
        month_start = dates[0]
        month_end = dates[-1]

        # 下载当月数据
        df = pro.daily_basic(
            start_date=month_start,
            end_date=month_end,
            fields=','.join(fields)
        )

        if df.empty:
            print(f"  {month}: 无数据")
            continue

        # 按日期保存
        for trade_dt in dates:
            day_df = df[df['trade_date'] == trade_dt]
            if day_df.empty:
                continue

            filename = f"daily_basic_{trade_dt}.parquet"
            filepath = output_path / filename

            table = pa.Table.from_pandas(day_df, preserve_index=False)
            pq.write_table(table, str(filepath))

        all_data.append(df)
        print(f"  {month}: {len(df)} 条记录")

        time.sleep(0.3)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n下载完成! 总计 {len(result)} 条记录")
        return result
    else:
        print("\n未下载到任何数据")
        return pd.DataFrame()


def download_all(
    start_date: str,
    end_date: str,
    daily_dir: str = DEFAULT_DAILY_DIR,
    daily_basic_dir: str = DEFAULT_DAILY_BASIC_DIR
):
    """
    下载所有数据（日线 + 每日基本面）

    Args:
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        daily_dir: 日线数据输出目录
        daily_basic_dir: 每日基本面输出目录
    """
    print("=" * 60)
    print("开始批量下载 Tushare 数据")
    print(f"时间范围: {start_date} ~ {end_date}")
    print("=" * 60)

    print("\n[1/2] 下载日线行情数据...")
    daily_df = download_daily(start_date, end_date, daily_dir)

    print("\n[2/2] 下载每日基本面数据...")
    basic_df = download_daily_basic(start_date, end_date, daily_basic_dir)

    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"  日线数据: {len(daily_df)} 条")
    print(f"  基本面数据: {len(basic_df)} 条")
    print("=" * 60)


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='从 Tushare 下载日线数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'data_type',
        choices=['daily', 'daily_basic', 'all'],
        help='数据类型: daily(日线), daily_basic(每日基本面), all(全部)'
    )

    parser.add_argument(
        '--start', '-s',
        required=True,
        help='开始日期，格式 YYYYMMDD'
    )

    parser.add_argument(
        '--end', '-e',
        required=True,
        help='结束日期，格式 YYYYMMDD'
    )

    parser.add_argument(
        '--daily-dir', '-d',
        default=DEFAULT_DAILY_DIR,
        help=f'日线数据输出目录 (默认: {DEFAULT_DAILY_DIR})'
    )

    parser.add_argument(
        '--basic-dir', '-b',
        default=DEFAULT_DAILY_BASIC_DIR,
        help=f'基本面数据输出目录 (默认: {DEFAULT_DAILY_BASIC_DIR})'
    )

    parser.add_argument(
        '--token', '-t',
        help='Tushare Token (可选，可通过环境变量或配置文件设置)'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置 Token
    if args.token:
        set_token(args.token)

    # 执行下载
    if args.data_type == 'daily':
        download_daily(args.start, args.end, args.daily_dir)
    elif args.data_type == 'daily_basic':
        download_daily_basic(args.start, args.end, args.basic_dir)
    elif args.data_type == 'all':
        download_all(args.start, args.end, args.daily_dir, args.basic_dir)


if __name__ == '__main__':
    main()
