"""
将季度财务数据转换为每日数据（历史数据版）
==========================================

功能：
- 将 cashflow/income/balance 季度数据转换为每日数据
- 使用季度数据前向填充（不依赖公告日期）
- 支持批量转换所有年份

使用说明：
---------

1. 转换全部三张表（2020-2025年）
   python convert_quarterly_to_daily.py --start 20200101 --end 20251231 --all

2. 只转换现金流量表
   python convert_quarterly_to_daily.py --start 20200101 --end 20251231 --cashflow

3. 只转换利润表
   python convert_quarterly_to_daily.py --start 20200101 --end 20251231 --income

4. 只转换资产负债表
   python convert_quarterly_to_daily.py --start 20200101 --end 20251231 --balance
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mylib.date_utils import parse_date as _parse_date
from mylib.date_utils import date_to_str as _date_to_str

# =============================================================================
# 配置
# =============================================================================

DATA_DIR = './daily_data/'
CASHFLOW_DIR = os.path.join(DATA_DIR, 'cashflow/')
CASHFLOW_DAILY_DIR = os.path.join(DATA_DIR, 'cashflow_daily/')
INCOME_DIR = os.path.join(DATA_DIR, 'income/')
INCOME_DAILY_DIR = os.path.join(DATA_DIR, 'income_daily/')
BALANCE_DIR = os.path.join(DATA_DIR, 'balance/')
BALANCE_DAILY_DIR = os.path.join(DATA_DIR, 'balance_daily/')

# 交易日历（用于确定每日数据范围）
TRADING_DAYS_FILE = os.path.join(DATA_DIR, 'trading_days.parquet')

# =============================================================================
# 工具函数
# =============================================================================

def get_all_trading_days(start_date: str, end_date: str) -> List[str]:
    """获取交易日列表"""
    # 方法1: 尝试从daily数据提取
    daily_dir = os.path.join(DATA_DIR, 'daily/')
    trading_days = set()

    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        year_str = str(year)
        # 从已下载的daily数据提取交易日
        for month in range(1, 13):
            month_str = f"{month:02d}"
            month_dir = Path(daily_dir) / year_str / month_str
            if month_dir.exists():
                for f in month_dir.glob('daily_*.parquet'):
                    date = f.name.replace('daily_', '').replace('.parquet', '')
                    if start_date <= date <= end_date:
                        trading_days.add(date)

    # 方法2: 如果没有daily数据，生成工作日
    if not trading_days:
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        current = start
        while current <= end:
            if current.weekday() < 5:  # 周一到周五
                trading_days.add(current.strftime('%Y%m%d'))
            current += timedelta(days=1)

    return sorted(trading_days)


def get_quarter_end_dates(start_date: str, end_date: str) -> List[str]:
    """获取季度末日期列表"""
    dates = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    for year in range(start_year, end_year + 1):
        # 季度末: 3月31日, 6月30日, 9月30日, 12月31日
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            date = f"{year}{month:02d}{day:02d}"
            if start_date <= date <= end_date:
                dates.append(date)

    return dates


parse_date = _parse_date
date_to_str = _date_to_str


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


# =============================================================================
# 核心转换函数
# =============================================================================

def load_quarterly_data(
    table_dir: str,
    table_name: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """加载季度财务数据"""
    print(f"  加载 {table_name} 季度数据...")

    all_data = []
    quarter_ends = get_quarter_end_dates(start_date, end_date)

    for q_date in quarter_ends:
        # 尝试多种目录结构
        possible_paths = [
            os.path.join(table_dir, q_date[:4], f"{int(q_date[4:6]):02d}", f"{table_name}_{q_date}.parquet"),
            os.path.join(table_dir, q_date[:4], q_date[4:6], f"{table_name}_{q_date}.parquet"),
            os.path.join(table_dir, q_date[:4], f"{table_name}_{q_date}.parquet"),
            os.path.join(table_dir, q_date, f"{table_name}_{q_date}.parquet"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df['quarter_date'] = q_date  # 添加季度标识
                all_data.append(df)
                print(f"    {q_date}: {len(df)} 条")
                break

    if not all_data:
        print(f"    未找到 {table_name} 数据")
        return pd.DataFrame()

    # 合并所有季度数据
    df = pd.concat(all_data, ignore_index=True)

    # 按股票和季度日期排序
    df = df.sort_values(['ts_code', 'quarter_date'])

    # 去重（保留最新的）
    df = df.drop_duplicates(subset=['ts_code', 'quarter_date'], keep='last')

    print(f"  合计: {len(df)} 条")
    return df


def convert_to_daily(
    df_quarterly: pd.DataFrame,
    trading_days: List[str],
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """将季度数据转换为每日数据

    Args:
        df_quarterly: 季度数据
        trading_days: 交易日列表
        fill_method: 填充方法 ('ffill'=前向填充, 'pad'=填充)

    Returns:
        每日数据
    """
    if df_quarterly.empty:
        return pd.DataFrame()

    print(f"  转换 {len(df_quarterly)} 条季度数据到 {len(trading_days)} 个交易日...")

    # 获取数值列（排除 ts_code, quarter_date）
    value_cols = [c for c in df_quarterly.columns if c not in ['ts_code', 'quarter_date']]

    # 获取所有股票
    ts_codes = df_quarterly['ts_code'].unique()
    print(f"  股票数: {len(ts_codes)}")

    # 创建日期范围
    trading_days_set = set(trading_days)

    # 准备结果
    all_daily = []

    # 逐个股票处理
    for ts_code in ts_codes:
        df_stock = df_quarterly[df_quarterly['ts_code'] == ts_code].copy()
        df_stock = df_stock.sort_values('quarter_date')

        if df_stock.empty:
            continue

        # 创建该股票的所有交易日记录
        stock_records = []
        for trade_date in trading_days:
            # 找到该日期之前的最新季度数据
            latest_quarter = None
            for _, row in df_stock.iterrows():
                if row['quarter_date'] <= trade_date:
                    latest_quarter = row
                else:
                    break

            if latest_quarter is not None:
                record = {'ts_code': ts_code, 'trade_date': trade_date}
                for col in value_cols:
                    record[col] = latest_quarter[col]
                stock_records.append(record)

        all_daily.extend(stock_records)

    if not all_daily:
        print(f"  无有效数据")
        return pd.DataFrame()

    df_daily = pd.DataFrame(all_daily)

    # 转换日期格式
    df_daily['trade_date'] = df_daily['trade_date'].astype(str).str.zfill(8).str[:8]

    print(f"  转换完成: {len(df_daily)} 条")
    return df_daily


def save_daily_data(
    df_daily: pd.DataFrame,
    output_dir: str,
    prefix: str,
    start_date: str,
    end_date: str
):
    """保存每日数据（按年/月结构）"""
    if df_daily.empty:
        return

    ensure_dir(output_dir)

    # 转换日期列
    df_daily['trade_date'] = df_daily['trade_date'].astype(str).str.zfill(8)

    # 按日期排序
    df_daily = df_daily.sort_values(['trade_date', 'ts_code'])

    # 去重
    df_daily = df_daily.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')

    # 保存每个日期的文件
    for trade_date in df_daily['trade_date'].unique():
        df_date = df_daily[df_daily['trade_date'] == trade_date]

        year = trade_date[:4]
        month = trade_date[4:6]

        output_path = os.path.join(output_dir, year, month)
        ensure_dir(output_path)

        filepath = os.path.join(output_path, f"{prefix}_{trade_date}.parquet")

        table = pa.Table.from_pandas(df_date, preserve_index=False)
        pq.write_table(table, filepath)


def save_annual_file(
    df_daily: pd.DataFrame,
    output_dir: str,
    year: str
):
    """保存年度合并文件"""
    if df_daily.empty:
        return

    output_file = os.path.join(output_dir, f"{year}_full.parquet")

    table = pa.Table.from_pandas(df_daily, preserve_index=False)
    pq.write_table(table, output_file)
    print(f"  保存 annual 文件: {output_file}")


# =============================================================================
# 主转换函数
# =============================================================================

def convert_table(
    table_name: str,
    table_dir: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    save_annual: bool = True
):
    """转换单个表"""
    print(f"\n{'='*60}")
    print(f"转换 {table_name} 数据: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    # 1. 获取交易日列表
    trading_days = get_all_trading_days(start_date, end_date)
    print(f"交易日数: {len(trading_days)}")

    # 2. 加载季度数据
    df_quarterly = load_quarterly_data(table_dir, table_name, start_date, end_date)

    if df_quarterly.empty:
        return

    # 3. 转换为每日数据
    df_daily = convert_to_daily(df_quarterly, trading_days)

    if df_daily.empty:
        return

    # 4. 保存每日文件
    print(f"\n保存每日文件...")
    prefix = f"{table_name}_daily"
    save_daily_data(df_daily, output_dir, prefix, start_date, end_date)

    # 5. 保存年度合并文件
    if save_annual:
        print(f"\n保存年度合并文件...")
        for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
            year_str = str(year)
            df_year = df_daily[df_daily['trade_date'].astype(str).str.startswith(year_str)]
            if not df_year.empty:
                save_annual_file(df_year, output_dir, year_str)

    print(f"\n✓ {table_name} 转换完成")


def convert_all(
    start_date: str,
    end_date: str,
    tables: List[str] = ['cashflow', 'income', 'balance'],
    save_annual: bool = True
):
    """转换所有表"""
    print(f"\n{'='*60}")
    print(f"开始转换: {start_date} ~ {end_date}")
    print(f"表: {tables}")
    print(f"{'='*60}")

    configs = {
        'cashflow': (CASHFLOW_DIR, CASHFLOW_DAILY_DIR),
        'income': (INCOME_DIR, INCOME_DAILY_DIR),
        'balance': (BALANCE_DIR, BALANCE_DAILY_DIR),
    }

    for table in tables:
        if table in configs:
            table_dir, output_dir = configs[table]
            convert_table(table, table_dir, output_dir, start_date, end_date, save_annual)


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='将季度财务数据转换为每日数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--start', '-s',
        required=True,
        help='开始日期 (YYYYMMDD)'
    )

    parser.add_argument(
        '--end', '-e',
        required=True,
        help='结束日期 (YYYYMMDD)'
    )

    parser.add_argument(
        '--cashflow',
        action='store_true',
        help='只转换现金流量表'
    )

    parser.add_argument(
        '--income',
        action='store_true',
        help='只转换利润表'
    )

    parser.add_argument(
        '--balance',
        action='store_true',
        help='只转换资产负债表'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='转换全部三张表'
    )

    parser.add_argument(
        '--no-annual',
        action='store_true',
        help='不保存年度合并文件'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 确定要转换的表
    if args.cashflow:
        tables = ['cashflow']
    elif args.income:
        tables = ['income']
    elif args.balance:
        tables = ['balance']
    elif args.all:
        tables = ['cashflow', 'income', 'balance']
    else:
        print("请指定 --cashflow, --income, --balance 或 --all")
        sys.exit(1)

    # 转换
    convert_all(
        start_date=args.start,
        end_date=args.end,
        tables=tables,
        save_annual=not args.no_annual
    )


if __name__ == '__main__':
    main()
