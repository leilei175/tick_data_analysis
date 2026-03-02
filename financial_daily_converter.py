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
from mylib.tushare_client import init_tushare as _init_tushare

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


def _get_trading_days_from_tushare(
    start_date: str,
    end_date: str,
    config_path: str = './config.py'
) -> List[str]:
    """从 Tushare 获取交易日列表，失败时回退到工作日。"""
    try:
        pro = _init_tushare(config_path=config_path)
        trade_cal = pro.trade_cal(
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            is_open='1'
        )
        if trade_cal is not None and not trade_cal.empty:
            return sorted(trade_cal['cal_date'].astype(str).tolist())
    except Exception as e:
        print(f"警告: 获取交易日失败，回退工作日历: {e}")

    start = parse_date(start_date)
    end = parse_date(end_date)
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(date_to_str(current))
        current += timedelta(days=1)
    return dates


def convert_table_to_daily_by_disclosure(
    table_name: str,
    start_date: str,
    end_date: str,
    data_dir: str,
    output_dir: str,
    *,
    source_file: Optional[str] = None,
    overwrite: bool = True,
    use_trading_days: bool = True
) -> int:
    """
    根据财报披露日期(优先f_ann_date,其次ann_date)生成每日财务数据并保存。

    输出文件命名:
    - cashflow_daily_YYYYMMDD.parquet
    - income_daily_YYYYMMDD.parquet
    - balance_daily_YYYYMMDD.parquet
    """
    prefix = f'{table_name}_daily'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if source_file is None:
        source_file = str(Path(data_dir) / f'{table_name}_all.parquet')
    source_path = Path(source_file)
    if not source_path.exists():
        raise FileNotFoundError(f"未找到季度全量文件: {source_path}")

    print(f"\n{'='*70}")
    print(f"按披露日期转换 {table_name}: {start_date} ~ {end_date}")
    print(f"来源: {source_path}")
    print(f"输出: {output_path}")
    print(f"{'='*70}")

    df = pq.read_table(source_path).to_pandas()
    if df.empty:
        print("来源数据为空，跳过")
        return 0

    required_cols = {'ts_code', 'end_date'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"{table_name} 缺少必要字段: {required_cols - set(df.columns)}")

    # 选择披露日期: 优先实际披露日期f_ann_date，其次ann_date
    if 'f_ann_date' in df.columns:
        disclosure = df['f_ann_date'].astype(str)
    else:
        disclosure = pd.Series([''] * len(df))
    if 'ann_date' in df.columns:
        ann = df['ann_date'].astype(str)
    else:
        ann = pd.Series([''] * len(df))

    disclosure = disclosure.where(disclosure.str.fullmatch(r'\d{8}'), ann)
    disclosure = disclosure.where(disclosure.str.fullmatch(r'\d{8}'))
    df['disclosure_date'] = disclosure
    df['end_date'] = df['end_date'].astype(str)

    # 过滤无效与区间外数据
    df = df[df['disclosure_date'].notna()].copy()
    df = df[(df['end_date'] >= '20100101') & (df['end_date'] <= end_date)]
    df = df[(df['disclosure_date'] >= '20100101') & (df['disclosure_date'] <= end_date)]
    df['disclosure_date_int'] = df['disclosure_date'].astype(int)
    if df.empty:
        print("有效披露数据为空，跳过")
        return 0

    # 同一股票同一披露日，保留报告期更晚的一条
    df = df.sort_values(['ts_code', 'disclosure_date', 'end_date'])
    df = df.drop_duplicates(subset=['ts_code', 'disclosure_date'], keep='last')

    # 输出列: ts_code, trade_date + 其他值列（排除原始日期/元字段）
    value_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLUMNS + ['disclosure_date', 'disclosure_date_int', 'ts_code', 'trade_date']
    ]
    keep_cols = ['ts_code', 'disclosure_date', 'disclosure_date_int'] + value_cols
    df = df[keep_cols].copy()

    if use_trading_days:
        all_days = _get_trading_days_from_tushare(start_date, end_date)
    else:
        all_days = get_date_list(start_date, end_date)
    if not all_days:
        print("日期列表为空，跳过")
        return 0

    # 逐年处理，降低内存峰值
    years = sorted({d[:4] for d in all_days})
    saved_files = 0

    for year in years:
        year_days = [d for d in all_days if d.startswith(year)]
        if not year_days:
            continue

        print(f"处理年份 {year}: {len(year_days)} 天")
        ts_codes = df['ts_code'].dropna().astype(str).unique()
        if len(ts_codes) == 0:
            continue

        # 构造当年全股票面板（trade_date x ts_code）
        left = pd.DataFrame({
            'trade_date': year_days * len(ts_codes),
            'ts_code': [code for code in ts_codes for _ in year_days]
        })
        left['trade_date_int'] = left['trade_date'].astype(int)

        right = df.sort_values(['disclosure_date_int', 'ts_code']).copy()
        left = left.sort_values(['trade_date_int', 'ts_code']).copy()

        year_panel = pd.merge_asof(
            left,
            right,
            left_on='trade_date_int',
            right_on='disclosure_date_int',
            by='ts_code',
            direction='backward'
        )
        year_panel = year_panel[year_panel['disclosure_date'].notna()].copy()
        if year_panel.empty:
            continue

        # 分日保存
        for trade_date, ddf in year_panel.groupby('trade_date', sort=True):
            out_file = output_path / f'{prefix}_{trade_date}.parquet'
            if out_file.exists() and not overwrite:
                continue
            # 去掉中间字段 disclosure_date，仅保留日频财报快照
            ddf = ddf.drop(columns=['disclosure_date', 'disclosure_date_int', 'trade_date_int'], errors='ignore')
            # 保证列顺序
            ordered = ['ts_code', 'trade_date'] + [c for c in value_cols if c not in ['ts_code']]
            ddf = ddf[[c for c in ordered if c in ddf.columns]]
            pq.write_table(pa.Table.from_pandas(ddf, preserve_index=False), str(out_file))
            saved_files += 1

    print(f"完成 {table_name}: 生成 {saved_files} 个每日文件")
    return saved_files


def convert_financial_daily_by_disclosure(
    start_date: str = '20150101',
    end_date: str = '20251231',
    tables: Optional[List[str]] = None,
    overwrite: bool = True
) -> Dict[str, int]:
    """批量按披露日期生成日频财务数据。"""
    if tables is None:
        tables = ['cashflow', 'income', 'balance']

    table_config = {
        'cashflow': (CASHFLOW_DIR, CASHFLOW_DAILY_DIR),
        'income': (INCOME_DIR, INCOME_DAILY_DIR),
        'balance': (BALANCE_DIR, BALANCE_DAILY_DIR),
    }

    results: Dict[str, int] = {}
    for table in tables:
        if table not in table_config:
            print(f"跳过未知表: {table}")
            continue
        data_dir, out_dir = table_config[table]
        count = convert_table_to_daily_by_disclosure(
            table_name=table,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            output_dir=out_dir,
            overwrite=overwrite,
            use_trading_days=True
        )
        results[table] = count
    return results


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
