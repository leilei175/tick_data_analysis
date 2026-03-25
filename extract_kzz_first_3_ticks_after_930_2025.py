"""
获取2025年所有可转债9:30之后前3个tick的数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import warnings
warnings.filterwarnings('ignore')

from mylib.get_tick_data import get_available_stocks

TICK_DIR = '/data1/quant-data/tick_2026/'

def process_kzz_first_3_ticks_after_930():
    """处理所有可转债9:30之后的前3个tick数据"""

    # 获取2025年有可转债数据的交易日
    dates = []
    data_dir = Path(TICK_DIR) / "2025"
    for month_dir in sorted(data_dir.iterdir()):
        if month_dir.is_dir():
            for day_dir in sorted(month_dir.iterdir()):
                if day_dir.is_dir():
                    files = list(day_dir.glob('11*.SH.parquet')) + list(day_dir.glob('12*.SH.parquet'))
                    if files:
                        try:
                            d = date(2025, int(month_dir.name), int(day_dir.name))
                            dates.append(d)
                        except:
                            pass

    print(f"找到 {len(dates)} 个有可转债数据的交易日")

    all_results = []

    for i, target_date in enumerate(dates):
        if i % 10 == 0:
            print(f"处理 {target_date} ({i+1}/{len(dates)})...")

        # 获取该日所有可转债
        try:
            all_stocks = get_available_stocks(dt=target_date, tick_dir=TICK_DIR)
        except:
            continue

        kzz_stocks = [s for s in all_stocks if s.startswith('11') or s.startswith('12')]

        if not kzz_stocks:
            continue

        for stock_code in kzz_stocks:
            try:
                date_path = Path(TICK_DIR) / f"{target_date.year:04d}" / f"{target_date.month:02d}" / f"{target_date.day:02d}"
                file_path = date_path / f"{stock_code}.parquet"
                if not file_path.exists():
                    continue

                df = pd.read_parquet(file_path)

                if 'time' not in df.columns:
                    continue

                dt_index = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
                df['time_str'] = dt_index.dt.strftime('%H:%M:%S.%f').str[:-3]

                # 筛选9:30之后的数据
                df_930 = df[df['time_str'] >= '09:30:00'].copy()

                if df_930.empty:
                    continue

                df_3 = df_930.head(3).copy()

                # 计算成交金额差值
                df_before = df[df['time_str'] < '09:30:00']
                if not df_before.empty:
                    last_amount_before = df_before['amount'].iloc[-1]
                    df_3['diff_amount'] = df_3['amount'] - last_amount_before
                else:
                    df_3['diff_amount'] = df_3['amount'].diff()
                    df_3.loc[df_3.index[0], 'diff_amount'] = np.nan

                # 计算涨跌幅
                df_3['pct_change'] = df_3['lastPrice'] / df_3['lastClose'] - 1

                df_3['tick_seq'] = range(1, len(df_3) + 1)
                df_3['stock_code'] = stock_code
                df_3['date'] = target_date

                cols = ['date', 'stock_code', 'tick_seq', 'time_str', 'amount', 'diff_amount', 'lastPrice', 'lastClose', 'pct_change']
                existing_cols = [c for c in cols if c in df_3.columns]
                df_output = df_3[existing_cols].rename(columns={'time_str': 'time'}).reset_index(drop=True)

                all_results.append(df_output)

            except Exception as e:
                continue

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = Path('kzz_first_3_ticks_after_930_2025.parquet')
        final_df.to_parquet(output_path, index=False)
        print(f"\n保存完成: {output_path}")
        print(f"总记录数: {len(final_df)}")
        print(f"可转债数量: {final_df['stock_code'].nunique()}")
        print(f"交易日数量: {final_df['date'].nunique()}")
    else:
        print("没有数据")

if __name__ == "__main__":
    process_kzz_first_3_ticks_after_930()
