#!/usr/bin/env python3
"""
将wind_hub目录下的parquet文件从：
- 行: 股票, 列: 时间
转换为：
- 行: 时间 (如2015q1), 列: 股票代码
"""
import os
import re
import pandas as pd
from pathlib import Path

# 季度映射
QUARTER_MAP = {
    '一季': 'q1',
    '中报': 'q2',
    '三季': 'q3',
    '年报': 'q4'
}

def parse_time_from_column(col_name: str) -> str:
    """从列名中提取时间并转换为 2021q1 格式"""
    # 匹配 [报告期] 后面的时间
    match = re.search(r'\[报告期\] (\d{4})(一季|中报|三季|年报)', col_name)
    if match:
        year = match.group(1)
        quarter = match.group(2)
        return f"{year}{QUARTER_MAP[quarter]}"
    return None

def convert_parquet(filepath: str):
    """转换单个parquet文件"""
    print(f"处理: {filepath}")

    # 读取数据
    df = pd.read_parquet(filepath)

    # 获取股票代码列
    stock_codes = df['证券代码'].values

    # 提取时间列并转换
    time_cols = {}
    for col in df.columns:
        if col in ['证券代码', '证券简称']:
            continue
        time_str = parse_time_from_column(col)
        if time_str:
            time_cols[col] = time_str

    # 只保留有时间数据的列
    df_data = df[list(time_cols.keys())]

    # 重命名列
    df_data.columns = [time_cols[c] for c in df_data.columns]

    # 转置：时间为index，股票代码为columns
    df_transposed = df_data.T
    df_transposed.columns = stock_codes

    # 保存到原文件（覆盖）
    output_path = filepath.replace('.parquet', '_converted.parquet')
    df_transposed.to_parquet(output_path)
    print(f"  保存为: {output_path}")
    print(f"  Shape: {df_transposed.shape} (时间 x 股票)")

def main():
    wind_hub_dir = Path('daily_data/wind_hub')

    parquet_files = list(wind_hub_dir.glob('*.parquet'))
    print(f"找到 {len(parquet_files)} 个parquet文件\n")

    for f in parquet_files:
        convert_parquet(str(f))

    print("\n完成!")

if __name__ == '__main__':
    main()
