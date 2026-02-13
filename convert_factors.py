"""
将因子数据转换为按因子单独存储的格式

输出格式：
- 文件命名：{prefix}_{factor_name}_{year}.parquet
- 数据格式：index=日期, columns=股票代码
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def convert_factors_to_wide_format(
    input_dir: str = "./factor/daily",
    output_dir: str = "./factor/by_factor",
    prefix: str = "zz1000"
):
    """
    将因子数据转换为宽格式（每个因子一个文件）

    Args:
        input_dir: 输入目录（包含原始因子文件）
        output_dir: 输出目录（保存转换后的因子文件）
        prefix: 文件前缀（如 zz1000）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有文件
    all_files = sorted(input_path.glob("factors_*.parquet"))

    if not all_files:
        print(f"未找到因子文件: {input_dir}")
        return

    # 读取并合并所有数据
    print("正在加载原始因子数据...")
    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            print(f"  已加载: {f.name} ({len(df):,} 行)")
        except Exception as e:
            print(f"  加载失败: {f.name} - {e}")

    if not dfs:
        print("没有有效数据可处理")
        return

    # 合并数据
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(['stock_code', 'datetime'])
    combined_df['date'] = pd.to_datetime(combined_df['date'])

    print(f"\n合并后数据: {len(combined_df):,} 行, "
          f"{combined_df['stock_code'].nunique()} 只股票, "
          f"{combined_df['date'].nunique()} 个交易日")

    # 找出因子列（排除非因子列）
    non_factor_cols = ['time', 'datetime', 'date', 'stock_code', 'lastPrice', 'volume', 'amount']
    factor_cols = [c for c in combined_df.columns if c not in non_factor_cols]

    print(f"发现 {len(factor_cols)} 个因子: {factor_cols}\n")

    # 按因子转换
    converted_files = []

    for factor in factor_cols:
        print(f"处理因子: {factor}")

        # 筛选有效数据
        factor_df = combined_df[['date', 'stock_code', factor]].dropna()

        if len(factor_df) == 0:
            print(f"  跳过 {factor}（无有效数据）")
            continue

        # 转换为宽格式（日期 x 股票）
        wide_df = factor_df.pivot_table(
            index='date',
            columns='stock_code',
            values=factor,
            aggfunc='mean'  # 如果同一天有多笔数据，取平均
        )

        # 按日期排序
        wide_df = wide_df.sort_index()

        # 按年份保存
        years = wide_df.index.year.unique()

        for year in years:
            year_df = wide_df[wide_df.index.year == year]

            # 生成文件名
            filename = f"{prefix}_{factor}_{year}.parquet"
            filepath = output_path / filename

            # 保存
            year_df.to_parquet(filepath)

            print(f"  保存: {filename} ({len(year_df)} 天, {year_df.shape[1]} 只股票)")
            converted_files.append({
                'filename': filename,
                'factor': factor,
                'year': year,
                'days': len(year_df),
                'stocks': year_df.shape[1],
                'size_mb': round(filepath.stat().st_size / (1024 * 1024), 2)
            })

    # 保存转换汇总
    summary_df = pd.DataFrame(converted_files)
    summary_path = output_path / "conversion_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("转换完成!")
    print(f"输出目录: {output_path}")
    print(f"转换汇总: {summary_path}")
    print(f"{'=' * 60}")

    return summary_df


def load_factor(factor_name: str, year: int, data_dir: str = "./factor/by_factor") -> pd.DataFrame:
    """
    加载单个因子数据

    Args:
        factor_name: 因子名称
        year: 年份
        data_dir: 数据目录

    Returns:
        DataFrame: index=日期, columns=股票代码
    """
    filepath = Path(data_dir) / f"zz1000_{factor_name}_{year}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    df = pd.read_parquet(filepath)
    return df


def list_available_factors(data_dir: str = "./factor/by_factor") -> pd.DataFrame:
    """
    列出所有可用的因子文件
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        return pd.DataFrame()

    files = list(data_path.glob("*.parquet"))

    if not files:
        return pd.DataFrame()

    info = []
    for f in files:
        # 解析文件名: zz1000_factor_name_year.parquet
        parts = f.stem.split('_')
        if len(parts) >= 4:
            prefix = parts[0]
            factor = '_'.join(parts[1:-1])  # 处理因子名中可能的下划线
            year = parts[-1]

            info.append({
                'filename': f.name,
                'prefix': prefix,
                'factor': factor,
                'year': int(year),
                'file_size_mb': round(f.stat().st_size / (1024 * 1024), 2)
            })

    return pd.DataFrame(info)


if __name__ == "__main__":
    print("因子数据格式转换工具")
    print("=" * 60)

    # 执行转换
    summary = convert_factors_to_wide_format(
        input_dir="./factor/daily",
        output_dir="./factor/by_factor",
        prefix="zz1000"
    )

    # 列出可用的因子文件
    print("\n可用因子文件列表:")
    print("-" * 60)
    available = list_available_factors()
    if not available.empty:
        for _, row in available.iterrows():
            print(f"  {row['filename']}")
            print(f"    因子: {row['factor']}, 年份: {row['year']}, 大小: {row['file_size_mb']} MB")
