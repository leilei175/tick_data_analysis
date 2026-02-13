"""
现金流量表季度数据转每日数据
=========================

功能：
- 将 cashflow 季度数据转换为每日数据
- 根据公告日期决定每天使用哪个季度的数据
- 保存为 Parquet 格式，与现有数据结构兼容

使用说明：
---------

1. 配置公告日期配置：
   - 在脚本中修改 ANNOUNCEMENT_DATES 字典
   - 格式: {'2025-10-10': '20250630', ...}
   - 键: 公告发布日期
   - 值: 对应的财报结束日期

2. 运行转换：

   # 转换 2025 年全年数据
   python cashflow_daily_converter.py --start 20250101 --end 20251231

   # 转换并下载缺失的季度数据
   python cashflow_daily_converter.py --start 20250101 --end 20251231 --fetch-missing

3. Python API 使用：

   from cashflow_daily_converter import convert_cashflow_to_daily, build_announcement_map

   # 构建公告日期映射
   announcement_map = build_announcement_map(cashflow_dir='./daily_data/cashflow')

   # 转换数据
   convert_cashflow_to_daily(
       start_date='20250101',
       end_date='20251231',
       cashflow_dir='./daily_data/cashflow',
       output_dir='./daily_data/cashflow_daily'
   )
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =============================================================================
# 配置
# =============================================================================

# 默认目录
CASHFLOW_DIR = './daily_data/cashflow/'
CASHFLOW_DAILY_DIR = './daily_data/cashflow_daily/'

# 公告日期配置
# 格式: 'YYYY-MM-DD': '财报结束日期'
# 表示在该日期发布财报，对应季度结束日期
# 财报发布日期由 announce_date 参数指定
ANNOUNCEMENT_DATES: Dict[str, str] = {
    '2025-02-28': '20241231',  # 2024年年报
    '2025-04-30': '20250331',  # 2025年一季报
    '2025-08-30': '20250630',  # 2025年中报
    '2025-10-10': '20250930',  # 2025年三季报
}

# 季度结束日期到下一个季度首日的映射
QUARTER_END_TO_NEXT_START = {
    '20240331': '20250401',  # Q1 -> Q2
    '20240630': '20250701',  # Q2 -> Q3
    '20240930': '20241001',  # Q3 -> Q4
    '20241231': '20250101',  # Q4 -> 明年 Q1
}

# 需要排除的财务指标列（保留 ts_code 和 date）
EXCLUDE_COLUMNS = ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type']

# =============================================================================
# 辅助函数
# =============================================================================

def parse_date(date_str: str) -> datetime:
    """解析日期字符串"""
    if isinstance(date_str, datetime):
        return date_str
    date_str = str(date_str).replace('-', '')
    return datetime.strptime(date_str, '%Y%m%d')


def get_quarter_from_date(date: datetime) -> str:
    """根据日期获取季度"""
    month = date.month
    if month <= 3:
        end_month = '0331'  # Q1
    elif month <= 6:
        end_month = '0630'  # Q2
    elif month <= 9:
        end_month = '0930'  # Q3
    else:
        end_month = '1231'  # Q4
    return f'{date.year}{end_month}'


def date_to_str(date: datetime, fmt: str = '%Y%m%d') -> str:
    """日期转字符串"""
    if isinstance(date, str):
        return date.replace('-', '')
    return date.strftime(fmt)


def get_trade_dates_between(start_date: str, end_date: str) -> List[str]:
    """
    获取交易日列表（暂用自然日，可替换为交易日历）
    实际使用时应接入 Tushare 或其他交易日历 API
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    dates = []
    current = start
    while current <= end:
        dates.append(date_to_str(current))
        current += timedelta(days=1)
    return dates


def parse_cashflow_filename(filename: str) -> Optional[str]:
    """从文件名解析季度结束日期"""
    match = re.match(r'cashflow_(\d{8})\.parquet', filename)
    if match:
        return match.group(1)
    return None


# =============================================================================
# 核心函数
# =============================================================================

def build_announcement_map(
    announcement_dates: Dict[str, str] = None,
    cashflow_dir: str = CASHFLOW_DIR
) -> Dict[str, str]:
    """
    构建公告日期到财报结束日期的映射

    Args:
        announcement_dates: 公告日期配置，格式 {'YYYY-MM-DD': 'YYYYMMDD'}
        cashflow_dir: cashflow 数据目录

    Returns:
        Dict[str, str]: {ann_date: end_date} 的映射
    """
    if announcement_dates is None:
        announcement_dates = ANNOUNCEMENT_DATES

    # 使用配置构建映射
    ann_map = {}
    for ann_date, end_date in announcement_dates.items():
        ann_map[ann_date.replace('-', '')] = end_date

    # 验证文件存在
    cashflow_path = Path(cashflow_dir)
    for end_date in set(ann_map.values()):
        expected_file = cashflow_path / f'cashflow_{end_date}.parquet'
        if not expected_file.exists():
            print(f"警告: 季度文件不存在 {expected_file}")
            # 尝试从 cashflow_all.parquet 提取
            print(f"  将尝试从 cashflow_all.parquet 提取数据")

    return ann_map


def load_quarterly_cashflow(
    end_date: str,
    cashflow_dir: str = CASHFLOW_DIR
) -> pd.DataFrame:
    """
    加载指定季度的现金流数据

    Args:
        end_date: 季度结束日期，如 '20250630'
        cashflow_dir: 数据目录

    Returns:
        pd.DataFrame: 季度数据
    """
    cashflow_path = Path(cashflow_dir)
    file_path = cashflow_path / f'cashflow_{end_date}.parquet'

    if file_path.exists():
        table = pq.read_table(file_path)
        return table.to_pandas()

    # 尝试从 cashflow_all.parquet 提取
    all_file = cashflow_path / 'cashflow_all.parquet'
    if all_file.exists():
        table = pq.read_table(all_file)
        df = table.to_pandas()
        return df[df['end_date'] == end_date]

    raise FileNotFoundError(f"无法找到季度数据: {end_date}")


def get_applicable_quarter(
    trade_date: str,
    announcement_map: Dict[str, str],
    quarter_files: Dict[str, pd.DataFrame]
) -> Tuple[str, pd.DataFrame]:
    """
    获取指定日期适用的季度数据

    Args:
        trade_date: 交易日期 YYYYMMDD
        announcement_map: 公告日期映射 {ann_date: end_date}
        quarter_files: 已加载的季度数据

    Returns:
        Tuple[str, pd.DataFrame]: (季度结束日期, 数据)
    """
    trade_dt = parse_date(trade_date)

    # 获取所有已发布的季度数据（公告日期 <= 交易日）
    applicable_quarters = []
    for ann_date, end_date in announcement_map.items():
        ann_dt = parse_date(ann_date)
        if ann_dt <= trade_dt:
            applicable_quarters.append((ann_dt, end_date))

    if not applicable_quarters:
        return None, pd.DataFrame()

    # 选择最新的季度
    latest = max(applicable_quarters, key=lambda x: x[0])
    end_date = latest[1]

    if end_date in quarter_files:
        return end_date, quarter_files[end_date]

    # 加载季度数据
    df = load_quarterly_cashflow(end_date)
    quarter_files[end_date] = df
    return end_date, df


def convert_cashflow_to_daily(
    start_date: str,
    end_date: str,
    cashflow_dir: str = CASHFLOW_DIR,
    output_dir: str = CASHFLOW_DAILY_DIR,
    announcement_dates: Dict[str, str] = None,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    将季度现金流数据转换为每日数据

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        cashflow_dir: 季度数据目录
        output_dir: 输出目录
        announcement_dates: 公告日期配置
        chunk_size: 分块大小

    Returns:
        pd.DataFrame: 转换后的每日数据
    """
    print(f"开始转换现金流数据: {start_date} ~ {end_date}")

    # 1. 加载所有公告日期配置
    announcement_map = build_announcement_map(announcement_dates, cashflow_dir)
    print(f"公告日期配置: {announcement_map}")

    # 2. 预加载所有季度数据
    print("加载季度数据...")
    quarter_files = {}
    for end_date in set(announcement_map.values()):
        try:
            df = load_quarterly_cashflow(end_date, cashflow_dir)
            quarter_files[end_date] = df
            print(f"  {end_date}: {len(df)} 条记录")
        except FileNotFoundError as e:
            print(f"  {end_date}: {e}")

    if not quarter_files:
        raise ValueError("未找到任何季度数据文件")

    # 3. 获取交易日列表
    trade_dates = get_trade_dates_between(start_date, end_date)
    print(f"日期范围: {len(trade_dates)} 天")

    # 4. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 5. 获取财务指标列
    sample_df = list(quarter_files.values())[0]
    value_columns = [c for c in sample_df.columns if c not in EXCLUDE_COLUMNS]
    print(f"财务指标列: {value_columns}")

    # 6. 按月分组处理并保存
    all_data = []
    months = {}

    for date in trade_dates:
        month = date[:6]
        if month not in months:
            months[month] = []
        months[month].append(date)

    print(f"按 {len(months)} 个月处理...")

    for month, dates in sorted(months.items()):
        print(f"  处理 {month}...")

        for trade_date in dates:
            end_date_q, quarter_df = get_applicable_quarter(
                trade_date, announcement_map, quarter_files
            )

            if quarter_df.empty:
                continue

            # 添加交易日
            daily_df = quarter_df.copy()
            daily_df['trade_date'] = trade_date

            # 保存每日文件
            filename = f"cashflow_daily_{trade_date}.parquet"
            filepath = output_path / filename

            # 只保留需要的列
            keep_cols = ['ts_code', 'trade_date'] + value_columns
            daily_df = daily_df[[c for c in keep_cols if c in daily_df.columns]]

            table = pa.Table.from_pandas(daily_df, preserve_index=False)
            pq.write_table(table, str(filepath))

            all_data.append(daily_df)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n转换完成! 总计 {len(result)} 条记录")
        return result
    else:
        print("\n未转换任何数据")
        return pd.DataFrame()


def update_cashflow_daily(
    start_date: str,
    end_date: str,
    announcement_date: str,
    quarter_end_date: str,
    cashflow_dir: str = CASHFLOW_DIR,
    cashflow_daily_dir: str = CASHFLOW_DAILY_DIR
):
    """
    更新指定公告日期之后的每日数据

    当新财报发布时，使用此函数更新后续数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        announcement_date: 财报公告日期 YYYYMMDD
        quarter_end_date: 财报结束日期 YYYYMMDD
        cashflow_dir: 季度数据目录
        cashflow_daily_dir: 每日数据目录
    """
    print(f"更新现金流每日数据: {start_date} ~ {end_date}")
    print(f"新财报: 公告日期={announcement_date}, 季度={quarter_end_date}")

    # 加载新季度数据
    new_quarter_df = load_quarterly_cashflow(quarter_end_date, cashflow_dir)
    print(f"新季度数据: {len(new_quarter_df)} 条")

    # 更新公告日期映射
    announcement_map = build_announcement_map()
    announcement_map[announcement_date] = quarter_end_date

    # 更新每日数据
    output_path = Path(cashflow_daily_dir)
    trade_dates = get_trade_dates_between(start_date, end_date)

    ann_dt = parse_date(announcement_date)

    for trade_date in trade_dates:
        trade_dt = parse_date(trade_date)

        # 只更新公告日期之后的数据
        if trade_dt < ann_dt:
            continue

        # 检查是否已有数据
        filename = f"cashflow_daily_{trade_date}.parquet"
        filepath = output_path / filename

        # 使用新季度数据
        daily_df = new_quarter_df.copy()
        daily_df['trade_date'] = trade_date

        # 保存
        value_columns = [c for c in daily_df.columns if c not in EXCLUDE_COLUMNS]
        keep_cols = ['ts_code', 'trade_date'] + value_columns
        daily_df = daily_df[[c for c in keep_cols if c in daily_df.columns]]

        table = pa.Table.from_pandas(daily_df, preserve_index=False)
        pq.write_table(table, str(filepath))

    print(f"更新完成: {len(trade_dates)} 天")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='将现金流季度数据转换为每日数据',
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
        '--cashflow-dir', '-c',
        default=CASHFLOW_DIR,
        help=f'季度数据目录 (默认: {CASHFLOW_DIR})'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default=CASHFLOW_DAILY_DIR,
        help=f'输出目录 (默认: {CASHFLOW_DAILY_DIR})'
    )

    parser.add_argument(
        '--ann-date',
        action='append',
        dest='ann_dates',
        help='公告日期配置，格式 YYYYMMDD:YYYYMMDD (可多次指定)'
    )

    parser.add_argument(
        '--update',
        action='store_true',
        help='仅更新模式，不重新生成所有数据'
    )

    parser.add_argument(
        '--fetch-missing',
        action='store_true',
        help='自动下载缺失的季度数据'
    )

    return parser.parse_args()


def parse_ann_dates(ann_dates_str: List[str]) -> Dict[str, str]:
    """解析公告日期参数"""
    result = {}
    for item in ann_dates_str:
        if ':' in item:
            ann_date, end_date = item.split(':')
            result[ann_date] = end_date
    return result


def main():
    """主函数"""
    args = parse_args()

    # 解析公告日期配置
    announcement_dates = None
    if args.ann_dates:
        announcement_dates = parse_ann_dates(args.ann_dates)
        print(f"使用自定义公告日期: {announcement_dates}")
    else:
        print(f"使用默认公告日期配置")
        print(f"  {ANNOUNCEMENT_DATES}")

    if args.update:
        # 更新模式
        if not announcement_dates:
            print("错误: 更新模式需要指定公告日期 --ann-date")
            sys.exit(1)

        # 假设季度结束日期与最近公告日期对应
        ann_date = list(announcement_dates.keys())[0]
        update_cashflow_daily(
            args.start, args.end,
            announcement_date=ann_date,
            quarter_end_date=announcement_dates[ann_date],
            cashflow_dir=args.cashflow_dir,
            cashflow_daily_dir=args.output_dir
        )
    else:
        # 完整转换
        convert_cashflow_to_daily(
            args.start, args.end,
            cashflow_dir=args.cashflow_dir,
            output_dir=args.output_dir,
            announcement_dates=announcement_dates
        )


if __name__ == '__main__':
    main()
