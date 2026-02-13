"""
计算未来收益率（简化版，内存友好）
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_returns():
    """计算并保存未来收益率"""
    input_dir = Path("./factor/daily")
    output_dir = Path("./factor/by_factor")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只使用zz1000_factors文件
    files = sorted(input_dir.glob("zz1000_factors_*.parquet"))
    if not files:
        print("未找到因子文件")
        return

    # 收集每天的收盘价数据（只保留 stock_code 和 lastPrice）
    print("加载价格数据...")
    daily_prices = {}
    for f in files:
        df = pd.read_parquet(f)
        date = str(df['date'].iloc[0])
        daily_prices[date] = df[['stock_code', 'lastPrice']].drop_duplicates('stock_code').set_index('stock_code')['lastPrice']
        print(f"  {date}: {len(daily_prices[date])} 只股票")

    dates = sorted(daily_prices.keys())
    print(f"\n共 {len(dates)} 个交易日")

    # 计算收益率
    print("计算收益率...")

    for period, outfile in [(1, 'return_1d.parquet'), (5, 'return_5d.parquet'), (10, 'return_10d.parquet')]:
        results = []
        for i in range(len(dates) - period):
            current = daily_prices[dates[i]]
            future = daily_prices[dates[i + period]]

            # 找到共同的股票
            common = current.index.intersection(future.index)
            if len(common) > 0:
                ret = (future.loc[common] / current.loc[common] - 1).reset_index()
                ret.columns = ['stock_code', f'return_{period}d']
                ret['date'] = dates[i]
                results.append(ret)

        if results:
            result_df = pd.concat(results, ignore_index=True)
            output_path = output_dir / outfile
            result_df.to_parquet(output_path, engine='pyarrow')
            print(f"  {outfile}: {len(result_df)} 行")

    print("\n完成！")


if __name__ == "__main__":
    calculate_returns()
