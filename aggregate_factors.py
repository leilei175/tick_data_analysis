"""
按天聚合高频因子并保存为宽格式
每个因子保存为一个parquet文件: index=date, columns=stock_code
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 因子列表
FACTORS = [
    'bid_ask_spread',
    'vwap_deviation',
    'trade_imbalance',
    'order_imbalance',
    'depth_imbalance',
    'realized_volatility',
    'effective_spread',
    'micro_price',
    'price_momentum',
    'trade_flow_intensity'
]

# 聚合方式: 'mean' 或 'last'
AGG_METHOD = 'mean'


def aggregate_factors(input_dir: str, output_dir: str, start_date: str = None, end_date: str = None):
    """
    按天聚合因子数据

    Args:
        input_dir: 原始因子数据目录
        output_dir: 输出目录
        start_date: 起始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找因子文件
    pattern = "zz1000_factors_*.parquet"
    if start_date or end_date:
        files = []
        for f in sorted(input_path.glob(pattern)):
            date_str = f.stem.split('_')[-1]
            if len(date_str) == 8:
                if start_date and date_str < start_date:
                    continue
                if end_date and date_str > end_date:
                    continue
                files.append(f)
    else:
        files = sorted(input_path.glob(pattern))

    if not files:
        print(f"未找到因子文件: {input_path / pattern}")
        return

    print(f"找到 {len(files)} 个因子文件")
    print(f"日期范围: {files[0].stem} -> {files[-1].stem}")

    # 按天聚合因子
    daily_factors = {f: {} for f in FACTORS}

    for filepath in files:
        date_str = filepath.stem.split('_')[-1]
        print(f"处理: {date_str}...", end=" ")

        try:
            df = pd.read_parquet(filepath)

            # 确保date列存在
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')

            # 聚合每个因子
            for factor in FACTORS:
                if factor not in df.columns:
                    print(f"\n  警告: {factor} 不在数据中")
                    continue

                if AGG_METHOD == 'mean':
                    aggregated = df.groupby('stock_code')[factor].mean()
                else:
                    aggregated = df.groupby('stock_code')[factor].last()

                daily_factors[factor][date_str] = aggregated

            print(f"OK ({len(df)} ticks)")

        except Exception as e:
            print(f"错误: {e}")
            continue

    # 保存每个因子到单独的parquet文件
    print("\n保存因子文件...")
    for factor in FACTORS:
        if not daily_factors[factor]:
            print(f"  {factor}: 无数据")
            continue

        # 合并所有日期
        factor_df = pd.DataFrame(daily_factors[factor])

        # 转置: index=date, columns=stock_code
        factor_df = factor_df.T.sort_index()

        # 确保列名是字符串
        factor_df.columns = factor_df.columns.astype(str)

        # 保存
        output_file = output_path / f"zz1000_{factor}.parquet"
        factor_df.to_parquet(output_file, engine='pyarrow')

        print(f"  {factor}: {factor_df.shape[0]} 天 x {factor_df.shape[1]} 股票")

    print("\n完成!")


def generate_yearly_files(output_dir: str, year: int = 2026):
    """
    生成年度版本的因子文件
    将每年数据单独保存
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for factor in FACTORS:
        yearly_file = output_path / f"zz1000_{factor}_{year}.parquet"

        # 查找所有年份文件
        all_files = sorted(output_path.glob(f"zz1000_{factor}_*.parquet"))

        dfs = []
        for f in all_files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                print(f"  读取 {f} 错误: {e}")

        if dfs:
            combined = pd.concat(dfs)
            combined.to_parquet(yearly_file, engine='pyarrow')
            print(f"  {factor}_{year}: {combined.shape[0]} 天")


def check_factors(output_dir: str):
    """检查因子文件"""
    output_path = Path(output_dir)

    print("因子文件检查:")
    print("-" * 60)

    for factor in FACTORS:
        files = sorted(output_path.glob(f"zz1000_{factor}*.parquet"))

        if not files:
            print(f"  {factor}: 不存在")
            continue

        for f in files:
            try:
                df = pd.read_parquet(f)
                print(f"  {f.name}: {df.shape[0]} 天 x {df.shape[1]} 股票")
                print(f"    日期: {df.index[0]} -> {df.index[-1]}")
                print(f"    样本: {df.iloc[0, :3].values}")
            except Exception as e:
                print(f"  {f.name}: 读取错误 - {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='聚合高频因子')
    parser.add_argument('--input', '-i', default='./factor/daily',
                        help='原始因子数据目录')
    parser.add_argument('--output', '-o', default='./factor/by_factor',
                        help='输出目录')
    parser.add_argument('--start', '-s', default=None,
                        help='起始日期 (YYYYMMDD)')
    parser.add_argument('--end', '-e', default=None,
                        help='结束日期 (YYYYMMDD)')
    parser.add_argument('--check', '-c', action='store_true',
                        help='检查现有因子文件')

    args = parser.parse_args()

    if args.check:
        check_factors(args.output)
    else:
        aggregate_factors(args.input, args.output, args.start, args.end)
