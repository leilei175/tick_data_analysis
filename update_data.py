"""
数据更新脚本
============

功能：
- 自动检测 daily_data 目录下各类数据的最新日期
- 下载更新日线数据、每日基本面数据
- 根据公告日期更新财务报表数据
- 支持增量更新和全量更新

使用说明：
---------

1. 命令行运行：

   # 更新所有数据（自动检测最新日期）
   python update_data.py

   # 只更新日线数据
   python update_data.py --daily

   # 只更新基本面数据
   python update_data.py --daily-basic

   # 只更新财务报表数据
   python update_data.py --financial

   # 指定更新日期范围
   python update_data.py --start 20260210 --end 20260211

   # 强制更新到今天（18:00后）
   python update_data.py --include-today

2. Python API：

   from update_data import update_all_data, update_daily_data

   # 更新所有数据
   update_all_data(include_today=False)

   # 只更新日线数据
   update_daily_data(start_date='20260210', end_date='20260211')
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Tushare
try:
    import tushare as ts
    HAS_TUSHARE = True
except ImportError:
    HAS_TUSHARE = False
    print("警告: 未安装 tushare")

# =============================================================================
# 配置
# =============================================================================

# 使用绝对路径（相对于脚本所在目录）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, 'daily_data')

# 数据目录配置
DATA_DIRS = {
    'daily': os.path.join(DEFAULT_DATA_DIR, 'daily/'),
    'daily_basic': os.path.join(DEFAULT_DATA_DIR, 'daily_basic/'),
    'cashflow_daily': os.path.join(DEFAULT_DATA_DIR, 'cashflow_daily/'),
    'income_daily': os.path.join(DEFAULT_DATA_DIR, 'income_daily/'),
    'balance_daily': os.path.join(DEFAULT_DATA_DIR, 'balance_daily/'),
}

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

# 当前日期（模拟2026年2月11日）
TODAY = datetime(2026, 2, 11).date()

# =============================================================================
# Token 管理
# =============================================================================

def get_token_from_config(config_path: str = None) -> str:
    """从配置文件读取 Token"""
    if config_path is None:
        # 默认使用脚本所在目录下的 config.py
        config_path = os.path.join(_SCRIPT_DIR, 'config.py')
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config.tushare_tk
    except Exception as e:
        raise ValueError(f"无法从 {config_path} 读取 Token: {e}")


def init_tushare():
    """初始化 Tushare"""
    if not HAS_TUSHARE:
        raise ImportError("未安装 tushare")

    token = get_token_from_config()
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


# =============================================================================
# 日期处理函数
# =============================================================================

def parse_date(date_str: str) -> datetime:
    """解析日期字符串"""
    date_str = str(date_str).replace('-', '').strip()
    return datetime.strptime(date_str, '%Y%m%d')


def date_to_str(date: datetime, fmt: str = '%Y%m%d') -> str:
    """日期转字符串"""
    return date.strftime(fmt)


def get_today_str() -> str:
    """获取今天日期字符串"""
    return TODAY.strftime('%Y%m%d')


def is_after_market_close() -> bool:
    """判断是否已收盘（18:00后）"""
    now = datetime.now()
    return now.hour >= 18


# =============================================================================
# 数据文件检测函数
# =============================================================================

def get_latest_date_from_dir(
    data_dir: str,
    prefix: str,
    pattern: str = None
) -> Optional[str]:
    """
    从目录中获取最新数据日期

    Args:
        data_dir: 数据目录
        prefix: 文件前缀 (如 'daily', 'cashflow_daily')
        pattern: 文件名模式正则表达式

    Returns:
        最新日期字符串，格式 YYYYMMDD，如果没有数据返回 None
    """
    path = Path(data_dir)

    if not path.exists():
        return None

    # 如果是按年/月组织的目录结构
    year_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.isdigit()]
    if year_dirs:
        latest_date = None
        for year_dir in sorted(year_dirs, reverse=True):
            month_dirs = [d for d in year_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            for month_dir in sorted(month_dirs, reverse=True):
                files = list(month_dir.glob(f'{prefix}_*.parquet'))
                if files:
                    for f in sorted(files, reverse=True):
                        date = parse_date_from_filename(f.name, prefix)
                        if date:
                            if latest_date is None or date > latest_date:
                                latest_date = date
                    # 找到数据就退出
                    if latest_date:
                        return latest_date
        return latest_date

    # 如果是扁平目录结构
    if pattern:
        regex = re.compile(pattern)
    else:
        regex = re.compile(rf'{prefix}_(\d{{8}})\.parquet$')

    latest_date = None
    for f in path.glob('*.parquet'):
        match = regex.match(f.name)
        if match:
            date_str = match.group(1)
            try:
                date = parse_date(date_str)
                if latest_date is None or date > latest_date:
                    latest_date = date
            except:
                pass

    if latest_date:
        return date_to_str(latest_date)
    return None


def parse_date_from_filename(filename: str, prefix: str) -> Optional[str]:
    """从文件名解析日期"""
    # 匹配 daily_YYYYMMDD.parquet 或 daily_YYYYMMDD_*.parquet
    patterns = [
        rf'{prefix}_(\d{{8}})\.parquet$',
        rf'{prefix}_(\d{{8}})_.*\.parquet$',
    ]

    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1)

    return None


def get_all_latest_dates() -> Dict[str, Optional[str]]:
    """获取所有数据类型的最新日期"""
    results = {}

    for data_type, data_dir in DATA_DIRS.items():
        prefix = data_type
        results[data_type] = get_latest_date_from_dir(data_dir, prefix)
        print(f"{data_type}: 最新数据日期 = {results[data_type]}")

    return results


# =============================================================================
# 交易日处理
# =============================================================================

def get_trade_dates(
    pro,
    start_date: str,
    end_date: str
) -> List[str]:
    """获取交易日列表"""
    trade_cal = pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1'
    )
    return trade_cal['cal_date'].tolist()


def get_next_trade_date(pro, current_date: str) -> Optional[str]:
    """获取下一个交易日"""
    current_dt = parse_date(current_date)
    end_date = date_to_str(current_dt + timedelta(days=10))

    trade_dates = get_trade_dates(pro, current_date, end_date)

    for td in trade_dates:
        if td > current_date:
            return td

    return None


# =============================================================================
# 数据下载函数
# =============================================================================

def download_daily_data(
    pro,
    start_date: str,
    end_date: str,
    output_dir: str = None
) -> pd.DataFrame:
    """下载日线数据

    Args:
        pro: Tushare Pro API
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录

    Returns:
        pd.DataFrame: 下载的数据
    """
    output_dir = output_dir or DATA_DIRS['daily']
    print(f"\n{'='*60}")
    print(f"下载日线数据: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    # 获取交易日列表
    trade_dates = get_trade_dates(pro, start_date, end_date)
    print(f"交易日数量: {len(trade_dates)}")

    if not trade_dates:
        print("没有交易日")
        return pd.DataFrame()

    # 过滤掉已存在的日期
    output_path = Path(output_dir)
    existing_dates = set()
    for year_dir in output_path.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and month_dir.name.isdigit():
                    for f in month_dir.glob('daily_*.parquet'):
                        date = parse_date_from_filename(f.name, 'daily')
                        if date:
                            existing_dates.add(date)

    # 只保留不存在的日期
    trade_dates = [d for d in trade_dates if d not in existing_dates]
    print(f"需要下载的新交易日: {len(trade_dates)}")

    if not trade_dates:
        print("所有日期的数据已存在")
        return pd.DataFrame()

    # 下载
    all_data = []
    for month, dates in _group_by_month(trade_dates):
        month_start = dates[0]
        month_end = dates[-1]

        df = pro.daily(
            start_date=month_start,
            end_date=month_end,
            fields=','.join(DAILY_FIELDS)
        )

        if not df.empty:
            # 按日期保存
            for trade_dt in dates:
                day_df = df[df['trade_date'] == trade_dt]
                if not day_df.empty:
                    _save_daily_file(day_df, output_dir, 'daily', trade_dt)

            all_data.append(df)
            print(f"  {month}: {len(df)} 条记录")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"✓ 日线数据下载完成: {len(result)} 条")
        return result

    return pd.DataFrame()


def download_daily_basic_data(
    pro,
    start_date: str,
    end_date: str,
    output_dir: str = None
) -> pd.DataFrame:
    """下载每日基本面数据

    Args:
        pro: Tushare Pro API
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录

    Returns:
        pd.DataFrame: 下载的数据
    """
    output_dir = output_dir or DATA_DIRS['daily_basic']
    print(f"\n{'='*60}")
    print(f"下载基本面数据: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    trade_dates = get_trade_dates(pro, start_date, end_date)
    print(f"交易日数量: {len(trade_dates)}")

    if not trade_dates:
        print("没有交易日")
        return pd.DataFrame()

    # 过滤掉已存在的日期
    output_path = Path(output_dir)
    existing_dates = set()
    for year_dir in output_path.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir() and month_dir.name.isdigit():
                    for f in month_dir.glob('daily_basic_*.parquet'):
                        date = parse_date_from_filename(f.name, 'daily_basic')
                        if date:
                            existing_dates.add(date)

    # 只保留不存在的日期
    trade_dates = [d for d in trade_dates if d not in existing_dates]
    print(f"需要下载的新交易日: {len(trade_dates)}")

    if not trade_dates:
        print("所有日期的数据已存在")
        return pd.DataFrame()

    all_data = []
    for month, dates in _group_by_month(trade_dates):
        month_start = dates[0]
        month_end = dates[-1]

        df = pro.daily_basic(
            start_date=month_start,
            end_date=month_end,
            fields=','.join(DAILY_BASIC_FIELDS)
        )

        if not df.empty:
            for trade_dt in dates:
                day_df = df[df['trade_date'] == trade_dt]
                if not day_df.empty:
                    _save_daily_file(day_df, output_dir, 'daily_basic', trade_dt)

            all_data.append(df)
            print(f"  {month}: {len(df)} 条记录")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"✓ 基本面数据下载完成: {len(result)} 条")
        return result

    return pd.DataFrame()


def _group_by_month(dates: List[str]) -> List[Tuple[str, List[str]]]:
    """将日期按月分组"""
    months = {}
    for d in dates:
        month = d[:6]
        if month not in months:
            months[month] = []
        months[month].append(d)

    return sorted(months.items())


def _save_daily_file(df: pd.DataFrame, output_dir: str, prefix: str, trade_date: str):
    """保存日线数据文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    year = trade_date[:4]
    month = trade_date[4:6]

    year_month_dir = output_path / year / month
    year_month_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{prefix}_{trade_date}.parquet'
    filepath = year_month_dir / filename

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(filepath))


# =============================================================================
# 财务报表数据更新
# =============================================================================

def update_financial_daily_data(
    start_date: str,
    end_date: str,
    tables: List[str] = None
):
    """
    更新财务报表每日数据

    财务报表数据使用公告日期机制更新：
    - 只有在财报发布后，相关日期的数据才会使用新财报
    - 需要先有季度数据，然后转换为每日数据
    """
    if tables is None:
        tables = ['cashflow', 'income', 'balance']

    print(f"\n{'='*60}")
    print(f"更新财务报表每日数据")
    print(f"{'='*60}")

    # 导入转换模块
    try:
        from financial_daily_converter import (
            convert_table_to_daily,
            convert_to_daily,
            update_to_latest_quarter
        )
    except ImportError:
        print("警告: 无法导入 financial_daily_converter 模块")
        return

    table_config = {
        'cashflow': (
            './daily_data/cashflow/',
            './daily_data/cashflow_daily/'
        ),
        'income': (
            './daily_data/income/',
            './daily_data/income_daily/'
        ),
        'balance': (
            './daily_data/balance/',
            './daily_data/balance_daily/'
        ),
    }

    for table in tables:
        if table not in table_config:
            continue

        data_dir, output_dir = table_config[table]

        print(f"\n处理 {table}...")

        # 检查是否有新的季度数据
        quarter_dir = Path(data_dir)
        quarter_files = list(quarter_dir.glob(f'{table}_*.parquet'))

        if not quarter_files:
            print(f"  没有季度数据源")
            continue

        # 使用转换函数更新
        convert_table_to_daily(
            table,
            start_date,
            end_date,
            data_dir,
            output_dir,
            skip_existing=True
        )


# =============================================================================
# 主更新函数
# =============================================================================

def update_daily_data(
    start_date: str = None,
    end_date: str = None,
    include_today: bool = False
):
    """
    更新日线和基本面数据

    Args:
        start_date: 开始日期，None 表示自动检测
        end_date: 结束日期，None 表示自动检测
        include_today: 是否包含今天的数据
    """
    pro = init_tushare()

    # 自动检测最新日期
    if start_date is None:
        latest_dates = get_all_latest_dates()
        latest_daily = latest_dates.get('daily')
        latest_basic = latest_dates.get('daily_basic')

        # 使用较晚的日期作为起始点
        if latest_daily and latest_basic:
            start_date = max(latest_daily, latest_basic)
        elif latest_daily:
            start_date = latest_daily
        elif latest_basic:
            start_date = latest_basic
        else:
            start_date = '20250101'

    # 确定结束日期
    if end_date is None:
        if include_today and is_after_market_close():
            end_date = get_today_str()
        else:
            # 获取前一个交易日
            end_date = get_today_str()
            trade_dates = get_trade_dates(pro, end_date, end_date)
            if trade_dates and trade_dates[-1] == end_date:
                # 今天是交易日
                pass
            else:
                # 今天不是交易日，获取最后一个交易日
                trade_dates = get_trade_dates(
                    pro,
                    date_to_str(parse_date(end_date) - timedelta(days=10)),
                    end_date
                )
                if trade_dates:
                    end_date = trade_dates[-1]

    print(f"\n更新范围: {start_date} ~ {end_date}")

    # 下载日线数据
    download_daily_data(pro, start_date, end_date)

    # 下载基本面数据
    download_daily_basic_data(pro, start_date, end_date)


def update_financial_data(
    start_date: str = None,
    end_date: str = None,
    tables: List[str] = None
):
    """
    更新财务报表每日数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        tables: 要更新的表列表
    """
    if tables is None:
        tables = ['cashflow', 'income', 'balance']

    if start_date is None:
        latest_dates = get_all_latest_dates()
        # 使用财务数据中最晚的日期
        latest_financial = max(
            latest_dates.get(t, '20250101')
            for t in ['cashflow_daily', 'income_daily', 'balance_daily']
        )
        start_date = latest_financial

    if end_date is None:
        end_date = get_today_str()

    update_financial_daily_data(start_date, end_date, tables)


def update_all_data(
    daily: bool = True,
    daily_basic: bool = True,
    financial: bool = True,
    start_date: str = None,
    end_date: str = None,
    include_today: bool = False
):
    """
    更新所有数据

    Args:
        daily: 是否更新日线数据
        daily_basic: 是否更新基本面数据
        financial: 是否更新财务报表数据
        start_date: 开始日期，None 表示自动检测
        end_date: 结束日期，None 表示自动检测
        include_today: 是否包含今天的数据（18:00后）
    """
    print("=" * 60)
    print("数据更新脚本")
    print("=" * 60)

    # 显示当前状态
    now = datetime.now()
    print(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"市场收盘后: {is_after_market_close()}")

    # 显示现有数据状态
    print("\n现有数据状态:")
    latest_dates = get_all_latest_dates()

    # 更新日线和基本面数据
    if daily or daily_basic:
        update_daily_data(
            start_date=start_date,
            end_date=end_date,
            include_today=include_today
        )

    # 更新财务报表数据
    if financial:
        update_financial_data(
            start_date=start_date,
            end_date=end_date
        )

    print("\n" + "=" * 60)
    print("数据更新完成!")
    print("=" * 60)


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='数据更新脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--daily',
        action='store_true',
        help='只更新日线数据'
    )

    parser.add_argument(
        '--daily-basic',
        action='store_true',
        help='只更新基本面数据'
    )

    parser.add_argument(
        '--financial',
        action='store_true',
        help='只更新财务报表数据'
    )

    parser.add_argument(
        '--start', '-s',
        help='开始日期 (YYYYMMDD)，默认自动检测'
    )

    parser.add_argument(
        '--end', '-e',
        help='结束日期 (YYYYMMDD)，默认自动检测'
    )

    parser.add_argument(
        '--include-today',
        action='store_true',
        help='包含今天的数据（仅在18:00后有效）'
    )

    parser.add_argument(
        '--no-financial',
        action='store_true',
        help='不更新财务报表数据'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 确定更新范围
    daily = args.daily or not (args.daily_basic or args.financial)
    daily_basic = args.daily_basic or not (args.daily or args.financial)
    financial = args.financial or not args.no_financial

    update_all_data(
        daily=daily,
        daily_basic=daily_basic,
        financial=financial,
        start_date=args.start,
        end_date=args.end,
        include_today=args.include_today
    )


if __name__ == '__main__':
    main()
