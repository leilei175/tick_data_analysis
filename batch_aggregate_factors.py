"""
批量聚合高频因子脚本
每日计算所有股票的10个高频指标，保存为宽格式
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
    'bid_ask_spread',      # 买卖价差
    'vwap_deviation',      # VWAP偏离度
    'trade_imbalance',      # 交易失衡度
    'order_imbalance',      # 订单失衡度
    'depth_imbalance',      # 深度失衡度
    'realized_volatility', # 已实现波动率
    'effective_spread',     # 有效价差
    'micro_price',          # 微价格
    'price_momentum',       # 价格动量
    'trade_flow_intensity'  # 交易流强度
]

# 数据源和输出配置
INPUT_DIR = "./factor/daily"
OUTPUT_DIR = "./factor/by_factor"


def get_all_tick_files():
    """获取所有tick数据文件"""
    input_path = Path(INPUT_DIR)

    # 优先使用zz1000文件（包含完整数据）
    files = sorted(input_path.glob("zz1000_factors_*.parquet"))

    if not files:
        # 回退到普通文件
        files = sorted(input_path.glob("factors_*.parquet"))

    return files


def aggregate_daily_factors(files, output_dir):
    """
    聚合每日因子数据

    Args:
        files: tick数据文件列表
        output_dir: 输出目录

    Returns:
        dict: 聚合结果统计
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 按天聚合因子
    daily_factors = {f: {} for f in FACTORS}
    date_info = []

    total_files = len(files)
    for idx, filepath in enumerate(files, 1):
        date_str = filepath.stem.split('_')[-1]  # 提取日期

        print(f"[{idx}/{total_files}] 处理 {date_str}...", end=" ")

        try:
            df = pd.read_parquet(filepath)

            # 确保date列存在
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')

            trade_date = df['date'].iloc[0]
            tick_count = len(df)
            stock_count = df['stock_code'].nunique()

            date_info.append({
                'date': trade_date,
                'ticks': tick_count,
                'stocks': stock_count
            })

            # 聚合每个因子（按股票取均值）
            for factor in FACTORS:
                if factor not in df.columns:
                    print(f"\n  警告: {factor} 不在数据中")
                    continue

                # 按股票代码分组，计算因子的日均值
                aggregated = df.groupby('stock_code')[factor].mean()
                daily_factors[factor][trade_date] = aggregated

            print(f"OK ({tick_count:,} ticks, {stock_count} stocks)")

        except Exception as e:
            print(f"错误: {e}")
            continue

    if not date_info:
        print("没有可处理的数据")
        return None

    # 保存每个因子到单独的parquet文件
    print("\n保存因子文件...")
    result = {
        'dates': [],
        'factors': {}
    }

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

        # 保存完整版本
        output_file = output_path / f"zz1000_{factor}.parquet"
        factor_df.to_parquet(output_file, engine='pyarrow')

        # 保存年度版本
        years = set(d.split('-')[0] for d in factor_df.index if isinstance(d, str))
        for year in years:
            yearly_file = output_path / f"zz1000_{factor}_{year}.parquet"
            year_dfs = []
            for d in factor_df.index:
                if isinstance(d, str) and d.startswith(year):
                    year_dfs.append(factor_df.loc[[d]])
            if year_dfs:
                pd.concat(year_dfs).to_parquet(yearly_file, engine='pyarrow')

        result['factors'][factor] = {
            'file': f"zz1000_{factor}.parquet",
            'shape': f"{factor_df.shape[0]} 天 x {factor_df.shape[1]} 股票",
            'date_range': f"{factor_df.index[0]} -> {factor_df.index[-1]}"
        }

        print(f"  {factor}: {factor_df.shape[0]} 天 x {factor_df.shape[1]} 股票")

    result['dates'] = date_info

    return result


def generate_report(result, output_file="因子数据生成报告.md"):
    """生成详细的处理报告"""
    if not result:
        return

    lines = [
        "# 高频因子数据生成报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## 数据概览",
        f"\n- **数据源**: {INPUT_DIR}",
        f"- **输出目录**: {OUTPUT_DIR}",
        f"- **处理交易日数**: {len(result['dates'])}",
        "\n### 每日数据统计",
        "| 日期 | Tick数 | 股票数 |",
        "|------|--------|--------|",
    ]

    for info in result['dates']:
        lines.append(f"| {info['date']} | {info['ticks']:,} | {info['stocks']} |")

    lines.extend([
        "\n## 因子文件清单",
        "\n| 因子名称 | 文件名 | 数据规模 | 日期范围 |",
        "|----------|--------|----------|----------|",
    ])

    for factor, info in result['factors'].items():
        lines.append(f"| {factor} | {info['file']} | {info['shape']} | {info['date_range']} |")

    lines.extend([
        "\n## 数据格式说明",
        "\n### 宽格式 (每因子一个文件)",
        "- **文件格式**: Parquet (Apache Arrow)",
        "- **行索引**: 日期 (YYYY-MM-DD)",
        "- **列索引**: 股票代码",
        "- **数据值**: 因子日均值",
        "\n### 示例数据结构:",
        "```",
        "                     000001.SZ  000002.SZ  000003.SZ",
        "2025-12-01          0.0105     0.0123     0.0098",
        "2025-12-02          0.0112     0.0118     0.0101",
        "```",
        "\n## 因子定义",
        "\n| 因子名称 | 英文名 | 说明 |",
        "|----------|--------|------|",
        "| 买卖价差 | bid_ask_spread | (askPrice - bidPrice) / midPrice |",
        "| VWAP偏离度 | vwap_deviation | (lastPrice - VWAP) / VWAP |",
        "| 交易失衡度 | trade_imbalance | (buyVolume - sellVolume) / totalVolume |",
        "| 订单失衡度 | order_imbalance | (askVol - bidVol) / (askVol + bidVol) |",
        "| 深度失衡度 | depth_imbalance | (askVol - bidVol) / totalDepth |",
        "| 已实现波动率 | realized_volatility | sqrt(sum(return^2)) |",
        "| 有效价差 | effective_spread | 2 * |lastPrice - midPrice| / midPrice |",
        "| 微价格 | micro_price | (askPrice*bidVol + bidPrice*askVol) / (askVol + bidVol) |",
        "| 价格动量 | price_momentum | return over lookback period |",
        "| 交易流强度 | trade_flow_intensity | trade_count / time |",
        "\n## 使用示例",
        "\n```python",
        "import pandas as pd",
        "\n# 读取因子数据",
        "factor_df = pd.read_parquet('./factor/by_factor/zz1000_order_imbalance.parquet')",
        "\n# 查看特定股票的因子值",
        "stock_factor = factor_df['000001.SZ']",
        "\n# 查看特定日期的所有因子值",
        "date_factors = factor_df.loc['2025-12-01']",
        "```",
        "\n## 相关文件",
        "\n- **收益率数据**: return_1d.parquet, return_5d.parquet, return_10d.parquet",
        "- **API接口**: /factor_dashboard/app.py",
    ])

    # 写入报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n报告已保存: {output_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("高频因子数据批量聚合")
    print("=" * 60)

    # 获取文件列表
    files = get_all_tick_files()
    print(f"\n找到 {len(files)} 个tick数据文件")

    if not files:
        print("未找到数据文件")
        sys.exit(1)

    # 聚合因子数据
    result = aggregate_daily_factors(files, OUTPUT_DIR)

    if result:
        # 生成报告
        generate_report(result)

    print("\n完成!")
