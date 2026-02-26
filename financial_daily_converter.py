"""
财务报表季度数据转每日数据
========================

功能：
- 将 cashflow/income/balance 季度数据转换为每日数据
- 根据公告日期决定每天使用哪个季度的数据
- 保存为 Parquet 格式

使用说明：
---------

1. 配置公告日期：
   - 在脚本中修改 ANNOUNCEMENT_DATES 配置
   - 格式: {'cashflow': {...}, 'income': {...}, 'balance': {...}}

2. 运行转换：

   # 转换全部三张表
   python financial_daily_converter.py --start 20250101 --end 20251231 --all

   # 只转换现金流量表
   python financial_daily_converter.py --start 20250101 --end 20251231 --cashflow

   # 只转换利润表
   python financial_daily_converter.py --start 20250101 --end 20251231 --income

   # 只转换资产负债表
   python financial_daily_converter.py --start 20250101 --end 20251231 --balance

3. Python API：

   from financial_daily_converter import (
       convert_to_daily,
       convert_cashflow_daily,
       convert_income_daily,
       convert_balance_daily
   )

   # 转换全部
   convert_to_daily(
       start_date='20250101',
       end_date='20251231',
       tables=['cashflow', 'income', 'balance']
   )
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mylib.date_utils import parse_date as _parse_date
from mylib.date_utils import date_to_str as _date_to_str

# =============================================================================
# 配置
# =============================================================================

# 默认目录
DATA_DIR = './daily_data/'
CASHFLOW_DIR = os.path.join(DATA_DIR, 'cashflow/')
CASHFLOW_DAILY_DIR = os.path.join(DATA_DIR, 'cashflow_daily/')
INCOME_DIR = os.path.join(DATA_DIR, 'income/')
INCOME_DAILY_DIR = os.path.join(DATA_DIR, 'income_daily/')
BALANCE_DIR = os.path.join(DATA_DIR, 'balance/')
BALANCE_DAILY_DIR = os.path.join(DATA_DIR, 'balance_daily/')

# 公告日期配置
# 格式: {'表名': {'公告日期(YYYYMMDD)': '财报结束日期(YYYYMMDD)'}}
# 例如: '20251010' 发布 '20250930' 财报
ANNOUNCEMENT_DATES: Dict[str, Dict[str, str]] = {
    'cashflow': {
        # 2024年年报
        '20250228': '20241231',
        # 2025年一季报
        '20250430': '20250331',
        # 2025年中报
        '20250830': '20250630',
        # 2025年三季报
        '20251010': '20250930',
    },
    'income': {
        '20250228': '20241231',
        '20250430': '20250331',
        '20250830': '20250630',
        '20251010': '20250930',
    },
    'balance': {
        '20250228': '20241231',
        '20250430': '20250331',
        '20250830': '20250630',
        '20251030': '20250630',  # 资产负债表通常晚几天
        '20251010': '20250930',
    },
}

# 公告日期配置模板（用于生成历史数据配置）
ANNOUNCEMENT_MONTHS = {
    # 季度: (发布月份, 季度结束日)
    'Q1': ('04', '0331'),   # Q1 财报 4月发布
    'Q2': ('08', '0630'),   # Q2 财报 8月发布
    'Q3': ('10', '0930'),   # Q3 财报 10月发布
    'Q4': ('04+1', '1231'), # Q4 财报次年4月发布
}

# 需要排除的列
EXCLUDE_COLUMNS = ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag']

# =============================================================================
# 辅助函数
# =============================================================================

parse_date = _parse_date
date_to_str = _date_to_str


def get_date_list(start_date: str, end_date: str) -> List[str]:
    """获取日期列表（自然日）"""
    start = parse_date(start_date)
    end = parse_date(end_date)
    dates = []
    current = start
    while current <= end:
        dates.append(date_to_str(current))
        current += timedelta(days=1)
    return dates


def parse_filename_date(filename: str, prefix: str) -> Optional[str]:
    """从文件名解析日期"""
    match = re.match(rf'{prefix}_(\d{{8}})\.parquet$', filename)
    if match:
        return match.group(1)
    return None


def get_quarter_end_dates(data_dir: str, prefix: str) -> List[str]:
    """获取目录中所有季度文件的日期"""
    path = Path(data_dir)
    dates = []
    for f in path.glob(f'{prefix}_*.parquet'):
        date = parse_filename_date(f.name, prefix)
        if date:
            dates.append(date)
    return sorted(dates)


def auto_generate_announcement_dates(data_dir: str, prefix: str, start_year: int = 2015, end_year: int = 2025) -> Dict[str, str]:
    """
    自动生成公告日期配置
    基于财报发布时间规律自动推断
    """
    # 获取实际存在的季度日期
    existing_quarters = get_quarter_end_dates(data_dir, prefix)

    ann_dates = {}

    for quarter_end in existing_quarters:
        year = int(quarter_end[:4])
        month_day = quarter_end[4:]

        # 确定发布时间
        if month_day == '0331':  # Q1
            ann_date = f'{year}0430'
        elif month_day == '0630':  # Q2
            ann_date = f'{year}0831'
        elif month_day == '0930':  # Q3
            ann_date = f'{year}1031'
        elif month_day == '1231':  # Q4 (年报，次年4月发布)
            # 对于2024年年报，在2025年2月发布
            if year == 2024:
                ann_date = '20250228'
            else:
                ann_date = f'{year + 1}0430'
        else:
            continue

        ann_dates[ann_date] = quarter_end

    return ann_dates


# =============================================================================
# 数据加载
# =============================================================================

def load_quarterly_data(
    end_date: str,
    data_dir: str,
    prefix: str
) -> pd.DataFrame:
    """加载指定季度的数据"""
    file_path = Path(data_dir) / f'{prefix}_{end_date}.parquet'

    if file_path.exists():
        table = pq.read_table(file_path)
        return table.to_pandas()

    # 尝试从所有文件提取
    all_file = Path(data_dir) / f'{prefix}_all.parquet'
    if all_file.exists():
        table = pq.read_table(all_file)
        df = table.to_pandas()
        if 'end_date' in df.columns:
            result = df[df['end_date'] == end_date]
            if not result.empty:
                return result

    raise FileNotFoundError(f"无法找到数据: {prefix}_{end_date}")


def load_quarterly_data_by_date(
    target_date: str,
    data_dir: str,
    prefix: str,
    announcement_map: Dict[str, str]
) -> Tuple[str, pd.DataFrame]:
    """
    根据目标日期加载适用的季度数据

    Returns:
        Tuple[str, pd.DataFrame]: (季度结束日期, 数据)
    """
    target_dt = parse_date(target_date)

    # 找到所有公告日期 <= 目标日期的季度
    applicable = []
    for ann_date, end_date in announcement_map.items():
        ann_dt = parse_date(ann_date)
        if ann_dt <= target_dt:
            applicable.append((ann_dt, end_date))

    if not applicable:
        return None, pd.DataFrame()

    # 选择最新的季度
    latest = max(applicable, key=lambda x: x[0])
    end_date = latest[1]

    df = load_quarterly_data(end_date, data_dir, prefix)
    return end_date, df


# =============================================================================
# 核心转换函数
# =============================================================================

def convert_table_to_daily(
    table_name: str,
    start_date: str,
    end_date: str,
    data_dir: str,
    output_dir: str,
    announcement_dates: Dict[str, str] = None,
    skip_existing: bool = True,
    use_auto_ann: bool = True
) -> pd.DataFrame:
    """
    将单张表的季度数据转换为每日数据

    Args:
        table_name: 表名 (cashflow/income/balance)
        start_date: 开始日期
        end_date: 结束日期
        data_dir: 季度数据目录
        output_dir: 输出目录
        announcement_dates: 公告日期配置
        skip_existing: 跳过已存在的文件
        use_auto_ann: 是否自动生成公告日期
    """
    prefix = table_name
    print(f"\n{'='*60}")
    print(f"转换 {table_name} 数据: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    # 1. 获取公告日期配置
    if announcement_dates is None:
        if use_auto_ann:
            announcement_dates = auto_generate_announcement_dates(data_dir, prefix)
            print(f"自动生成的公告日期: {announcement_dates}")
        else:
            announcement_dates = ANNOUNCEMENT_DATES.get(table_name, {})
            print(f"使用配置的公告日期: {announcement_dates}")

    if not announcement_dates:
        print(f"警告: {table_name} 没有公告日期配置")
        return pd.DataFrame()

    # 2. 获取季度数据日期列表并加载
    print("加载季度数据...")
    quarter_files = {}
    for q_end_date in set(announcement_dates.values()):
        try:
            df = load_quarterly_data(q_end_date, data_dir, prefix)
            quarter_files[q_end_date] = df
            print(f"  {q_end_date}: {len(df)} 条")
        except FileNotFoundError as e:
            print(f"  {q_end_date}: 未找到")

    if not quarter_files:
        print("未找到任何季度数据")
        return pd.DataFrame()

    # 3. 获取日期列表
    dates = get_date_list(start_date, end_date)
    print(f"日期范围: {len(dates)} 天")

    # 4. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 5. 确定要保留的列
    sample_df = list(quarter_files.values())[0]
    value_columns = [c for c in sample_df.columns if c not in EXCLUDE_COLUMNS]
    print(f"保留列数: {len(value_columns)}")

    # 6. 按月分组处理
    all_data = []
    months = {}
    for d in dates:
        month = d[:6]
        if month not in months:
            months[month] = []
        months[month].append(d)

    print(f"按 {len(months)} 个月处理...")

    for month, month_dates in sorted(months.items()):
        print(f"  {month}...", end=' ', flush=True)

        for trade_date in month_dates:
            filename = f'{prefix}_daily_{trade_date}.parquet'
            filepath = output_path / filename

            # 跳过已存在的文件
            if skip_existing and filepath.exists():
                continue

            # 获取适用季度
            trade_dt = parse_date(trade_date)
            applicable = []
            for ann_date, q_end in announcement_dates.items():
                ann_dt = parse_date(ann_date)
                if ann_dt <= trade_dt:
                    applicable.append((ann_dt, q_end))

            if not applicable:
                continue

            latest = max(applicable, key=lambda x: x[0])
            q_end = latest[1]

            if q_end not in quarter_files:
                continue

            # 创建每日数据
            daily_df = quarter_files[q_end].copy()
            daily_df['trade_date'] = trade_date

            # 只保留需要的列
            keep_cols = ['ts_code', 'trade_date'] + value_columns
            daily_df = daily_df[[c for c in keep_cols if c in daily_df.columns]]

            # 保存
            table = pa.Table.from_pandas(daily_df, preserve_index=False)
            pq.write_table(table, str(filepath))

            all_data.append(daily_df)

        print(f"✓")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n完成! 总计 {len(result)} 条记录")
        return result
    else:
        print("\n未转换任何数据")
        return pd.DataFrame()


def convert_to_daily(
    start_date: str,
    end_date: str,
    tables: List[str] = None,
    skip_existing: bool = True
):
    """
    批量转换多张表的季度数据为每日数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        tables: 表名列表 ['cashflow', 'income', 'balance']
        skip_existing: 跳过已存在的文件
    """
    if tables is None:
        tables = ['cashflow', 'income', 'balance']

    print("="*60)
    print(f"批量转换财务报表为每日数据")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"表: {tables}")
    print("="*60)

    table_config = {
        'cashflow': (CASHFLOW_DIR, CASHFLOW_DAILY_DIR),
        'income': (INCOME_DIR, INCOME_DAILY_DIR),
        'balance': (BALANCE_DIR, BALANCE_DAILY_DIR),
    }

    results = {}
    for table in tables:
        if table not in table_config:
            print(f"未知表: {table}")
            continue

        data_dir, output_dir = table_config[table]
        results[table] = convert_table_to_daily(
            table, start_date, end_date,
            data_dir, output_dir,
            skip_existing=skip_existing
        )

    print("\n" + "="*60)
    print("全部转换完成!")
    for table, result in results.items():
        print(f"  {table}: {len(result)} 条")
    print("="*60)


def update_to_latest_quarter(
    table_name: str,
    announcement_date: str,
    quarter_end_date: str,
    start_date: str = None,
    end_date: str = None,
    data_dir: str = None,
    output_dir: str = None
):
    """
    当新财报发布后，更新每日数据使用最新季度数据

    Args:
        table_name: 表名
        announcement_date: 公告日期 (YYYYMMDD)
        quarter_end_date: 季度结束日期 (YYYYMMDD)
        start_date: 更新的开始日期
        end_date: 更新的结束日期
    """
    prefix = table_name

    if data_dir is None:
        data_dir = {
            'cashflow': CASHFLOW_DIR,
            'income': INCOME_DIR,
            'balance': BALANCE_DIR,
        }[table_name]

    if output_dir is None:
        output_dir = {
            'cashflow': CASHFLOW_DAILY_DIR,
            'income': INCOME_DAILY_DIR,
            'balance': BALANCE_DAILY_DIR,
        }[table_name]

    print(f"\n更新 {table_name} 每日数据")
    print(f"新财报: {quarter_end_date}, 公告日期: {announcement_date}")

    # 加载新季度数据
    new_quarter = load_quarterly_data(quarter_end_date, data_dir, prefix)
    print(f"新季度数据: {len(new_quarter)} 条")

    # 确定更新范围
    if start_date is None:
        start_date = announcement_date
    if end_date is None:
        end_date = announcement_date

    dates = get_date_list(start_date, end_date)
    output_path = Path(output_dir)

    ann_dt = parse_date(announcement_date)

    updated = 0
    for trade_date in dates:
        trade_dt = parse_date(trade_date)

        # 只更新公告日期之后的数据
        if trade_dt < ann_dt:
            continue

        filepath = output_path / f'{prefix}_daily_{trade_date}.parquet'

        # 创建每日数据
        daily_df = new_quarter.copy()
        daily_df['trade_date'] = trade_date

        # 只保留需要的列
        value_columns = [c for c in daily_df.columns if c not in EXCLUDE_COLUMNS]
        keep_cols = ['ts_code', 'trade_date'] + value_columns
        daily_df = daily_df[[c for c in keep_cols if c in daily_df.columns]]

        table = pa.Table.from_pandas(daily_df, preserve_index=False)
        pq.write_table(table, str(filepath))
        updated += 1

    print(f"更新完成: {updated} 天")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='将财务报表季度数据转换为每日数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
        '--skip',
        action='store_true',
        default=True,
        help='跳过已存在的文件 (默认)'
    )

    parser.add_argument(
        '--no-skip',
        action='store_false',
        dest='skip',
        help='不跳过已存在的文件'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 确定要转换的表
    tables = []
    if args.cashflow:
        tables.append('cashflow')
    if args.income:
        tables.append('income')
    if args.balance:
        tables.append('balance')
    if args.all or not tables:
        tables = ['cashflow', 'income', 'balance']

    convert_to_daily(
        args.start, args.end,
        tables=tables,
        skip_existing=args.skip
    )


if __name__ == '__main__':
    main()
