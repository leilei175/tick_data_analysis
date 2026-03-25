"""
获取2026年所有可转债前3个tick的数据，计算成交金额差值、lastPrice和涨跌幅
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import warnings
warnings.filterwarnings('ignore')

from mylib.get_tick_data import get_tick_data_short, get_available_stocks

TICK_DIR = '/data1/quant-data/tick_2026/'

def process_kzz_first_3_ticks():
    """处理所有可转债的前3个tick数据"""

    # 获取2026年所有交易日
    dates = []
    data_dir = Path(TICK_DIR) / "2026"
    if data_dir.exists():
        for month_dir in sorted(data_dir.iterdir()):
            if month_dir.is_dir():
                for day_dir in sorted(month_dir.iterdir()):
                    if day_dir.is_dir():
                        try:
                            d = date(2026, int(month_dir.name), int(day_dir.name))
                            dates.append(d)
                        except:
                            pass

    print(f"找到 {len(dates)} 个交易日")

    all_results = []

    for i, target_date in enumerate(dates):
        print(f"处理 {target_date} ({i+1}/{len(dates)})...")

        # 获取该日所有可转债（11开头和12开头的为可转债）
        try:
            all_stocks = get_available_stocks(dt=target_date, tick_dir=TICK_DIR)
        except:
            continue

        kzz_stocks = [s for s in all_stocks if s.startswith('11') or s.startswith('12')]

        if not kzz_stocks:
            continue

        # 批量获取这些可转债的tick数据
        try:
            result = get_tick_data_short(kzz_stocks, start_date=target_date, end_date=target_date, tick_dir=TICK_DIR)
        except Exception as e:
            print(f"  获取数据失败: {e}")
            continue

        if isinstance(result, dict):
            for stock_code, df in result.items():
                if df.empty:
                    continue

                # 获取前3个tick
                df_3 = df.head(3).copy()

                # 计算成交金额差值
                df_3['diff_amount'] = df_3['amount'].diff()

                # 计算涨跌幅
                if 'lastClose' in df_3.columns:
                    df_3['pct_change'] = df_3['lastPrice'] / df_3['lastClose'] - 1

                # 添加序号
                df_3['tick_seq'] = range(1, len(df_3) + 1)
                df_3['stock_code'] = stock_code
                df_3['date'] = target_date

                # 选择需要的列
                cols = ['date', 'stock_code', 'tick_seq', 'time', 'amount', 'diff_amount', 'lastPrice', 'lastClose', 'pct_change']
                existing_cols = [c for c in cols if c in df_3.columns]
                df_output = df_3[existing_cols].reset_index(drop=True)

                all_results.append(df_output)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = Path('kzz_first_3_ticks_2026.parquet')
        final_df.to_parquet(output_path, index=False)
        print(f"\n保存完成: {output_path}")
        print(f"总记录数: {len(final_df)}")
        print(f"可转债数量: {final_df['stock_code'].nunique()}")
        print(f"交易日数量: {final_df['date'].nunique()}")
        print("\n前10条数据:")
        print(final_df.head(10))
    else:
        print("没有数据")

if __name__ == "__main__":
    process_kzz_first_3_ticks()
