"""
分析可转债集合竞价与开盘后表现的关系
由于tick数据采样限制，集合竞价成交额以9:25:01的累计成交额为准
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import warnings
warnings.filterwarnings('ignore')

TICK_DIR = '/data1/quant-data/tick_2026/'

def read_kzz_tick(stock_code: str, target_date: date) -> pd.DataFrame:
    """读取可转债tick数据"""
    date_path = Path(TICK_DIR) / f"{target_date.year:04d}" / f"{target_date.month:02d}" / f"{target_date.day:02d}"
    file_path = date_path / f"{stock_code}.parquet"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(file_path)

    if 'time' in df.columns:
        dt_index = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        df['time_str'] = dt_index.dt.strftime('%H:%M:%S.%f').str[:-3]

    return df

def get_call_auction_data(df: pd.DataFrame) -> dict:
    """提取集合竞价和开盘数据"""
    if df.empty:
        return {}

    result = {}

    # 9:25:01 集合竞价成交额（累计）
    ca_925 = df[df['time_str'] == '09:25:01']
    if ca_925.empty:
        ca_925 = df[(df['time_str'] >= '09:25:00') & (df['time_str'] < '09:25:10')]

    # 9:30:01 开盘后成交额（累计）
    ca_930 = df[df['time_str'] == '09:30:01']
    if ca_930.empty:
        ca_930 = df[(df['time_str'] >= '09:30:00') & (df['time_str'] < '09:30:10')]

    # 9:35:00 的数据
    ca_935 = df[(df['time_str'] >= '09:35:00') & (df['time_str'] < '09:35:10')]

    # 10:00:00 的数据
    ca_1000 = df[(df['time_str'] >= '10:00:00') & (df['time_str'] < '10:00:10')]

    if not ca_925.empty:
        result['amount_ca'] = ca_925['amount'].iloc[-1]  # 集合竞价成交额
        result['vol_ca'] = ca_925['volume'].iloc[-1]
        result['lastClose'] = ca_925['lastClose'].iloc[-1]
        result['open'] = ca_925['lastPrice'].iloc[-1]
        if result['lastClose'] and result['lastClose'] > 0:
            result['ca_pct'] = (result['open'] - result['lastClose']) / result['lastClose'] * 100

    if not ca_930.empty:
        result['amount_930'] = ca_930['amount'].iloc[-1]
        result['price_930'] = ca_930['lastPrice'].iloc[-1]

    if not ca_935.empty:
        result['amount_935'] = ca_935['amount'].iloc[-1]
        result['price_935'] = ca_935['lastPrice'].iloc[-1]

    if not ca_1000.empty:
        result['amount_1000'] = ca_1000['amount'].iloc[-1]
        result['price_1000'] = ca_1000['lastPrice'].iloc[-1]

    return result

if __name__ == "__main__":
    # 分析2025年12月的数据
    dates = []
    for m in ['12']:
        for d in ['01', '02', '03', '04', '05', '08', '09', '10', '11', '12', '15', '16', '17', '18', '19', '22', '23', '24', '29', '30', '31']:
            try:
                dates.append(date(2025, int(m), int(d)))
            except:
                pass

    results = []
    for target_date in dates:
        date_path = Path(TICK_DIR) / f"{target_date.year:04d}" / f"{target_date.month:02d}" / f"{target_date.day:02d}"
        if not date_path.exists():
            continue
        kzz_files = list(date_path.glob("11*.SH.parquet")) + list(date_path.glob("12*.SH.parquet"))

        for f in kzz_files:
            stock_code = f.stem
            df = read_kzz_tick(stock_code, target_date)
            if df.empty:
                continue

            data = get_call_auction_data(df)
            if 'amount_ca' in data and 'open' in data and data['amount_ca'] > 0:
                data['stock_code'] = stock_code
                data['date'] = target_date
                results.append(data)

    df_result = pd.DataFrame(results)
    print(f"分析样本数: {len(df_result)}")

    if len(df_result) == 0:
        print("没有数据")
        exit()

    # 计算各时间段涨跌幅
    if 'price_930' in df_result.columns and 'open' in df_result.columns:
        df_result['pct_930'] = (df_result['price_930'] - df_result['open']) / df_result['open'] * 100

    if 'price_935' in df_result.columns and 'open' in df_result.columns:
        df_result['pct_935'] = (df_result['price_935'] - df_result['open']) / df_result['open'] * 100

    if 'price_1000' in df_result.columns and 'open' in df_result.columns:
        df_result['pct_1000'] = (df_result['price_1000'] - df_result['open']) / df_result['open'] * 100

    # 计算成交额变化
    if 'amount_930' in df_result.columns:
        df_result['amount_930_diff'] = df_result['amount_930'] - df_result['amount_ca']

    if 'amount_935' in df_result.columns:
        df_result['amount_935_diff'] = df_result['amount_935'] - df_result['amount_ca']

    # ========== 分析1: 集合竞价成交额 vs 集合竞价涨跌幅 ==========
    print("\n" + "="*60)
    print("【分析1】集合竞价成交额 vs 集合竞价涨跌幅")
    print("="*60)
    df_valid = df_result[df_result['ca_pct'].notna()].copy()

    if len(df_valid) > 5:
        df_valid['amount_q'] = pd.qcut(df_valid['amount_ca'], q=5, labels=['Q1(小)', 'Q2', 'Q3', 'Q4', 'Q5(大)'], duplicates='drop')
        print(f"样本数: {len(df_valid)}")
        print("\n按集合竞价成交额分组的涨跌幅:")
        print(df_valid.groupby('amount_q')['ca_pct'].agg(['mean', 'std', 'count']).round(3))
        corr = df_valid['amount_ca'].corr(df_valid['ca_pct'])
        print(f"\n相关性: {corr:.4f}")

    # ========== 分析2: 集合竞价成交额 vs 开盘后5分钟涨跌幅 ==========
    print("\n" + "="*60)
    print("【分析2】集合竞价成交额 vs 开盘到9:35涨跌幅")
    print("="*60)
    df_valid2 = df_result[df_result['pct_935'].notna()].copy()

    if len(df_valid2) > 5:
        df_valid2['amount_q'] = pd.qcut(df_valid2['amount_ca'], q=5, labels=['Q1(小)', 'Q2', 'Q3', 'Q4', 'Q5(大)'], duplicates='drop')
        print(f"样本数: {len(df_valid2)}")
        print("\n按集合竞价成交额分组的9:35涨跌幅:")
        print(df_valid2.groupby('amount_q')['pct_935'].agg(['mean', 'std', 'count']).round(3))
        corr2 = df_valid2['amount_ca'].corr(df_valid2['pct_935'])
        print(f"\n相关性: {corr2:.4f}")

    # ========== 分析3: 集合竞价成交额 vs 10:00涨跌幅 ==========
    print("\n" + "="*60)
    print("【分析3】集合竞价成交额 vs 开盘到10:00涨跌幅")
    print("="*60)
    df_valid3 = df_result[df_result['pct_1000'].notna()].copy()

    if len(df_valid3) > 5:
        df_valid3['amount_q'] = pd.qcut(df_valid3['amount_ca'], q=5, labels=['Q1(小)', 'Q2', 'Q3', 'Q4', 'Q5(大)'], duplicates='drop')
        print(f"样本数: {len(df_valid3)}")
        print("\n按集合竞价成交额分组的10:00涨跌幅:")
        print(df_valid3.groupby('amount_q')['pct_1000'].agg(['mean', 'std', 'count']).round(3))
        corr3 = df_valid3['amount_ca'].corr(df_valid3['pct_1000'])
        print(f"\n相关性: {corr3:.4f}")

    # ========== 分析4: 成交额放大倍数 ==========
    print("\n" + "="*60)
    print("【分析4】开盘后成交额放大倍数")
    print("="*60)
    if 'amount_930_diff' in df_result.columns:
        df_result['amplify_5min'] = df_result['amount_930_diff'] / df_result['amount_ca'].replace(0, np.nan)
        print(f"开盘5分钟成交额 / 集合竞价成交额:")
        print(f"  均值: {df_result['amplify_5min'].mean():.2f}x")
        print(f"  中位数: {df_result['amplify_5min'].median():.2f}x")

    if 'amount_935_diff' in df_result.columns:
        df_result['amplify_10min'] = df_result['amount_935_diff'] / df_result['amount_ca'].replace(0, np.nan)
        print(f"\n开盘10分钟成交额 / 集合竞价成交额:")
        print(f"  均值: {df_result['amplify_10min'].mean():.2f}x")
        print(f"  中位数: {df_result['amplify_10min'].median():.2f}x")

    # ========== 综合统计 ==========
    print("\n" + "="*60)
    print("【综合统计】")
    print("="*60)
    print(f"\n集合竞价涨跌幅:")
    print(f"  均值: {df_result['ca_pct'].mean():.3f}%")
    print(f"  标准差: {df_result['ca_pct'].std():.3f}%")
    print(f"  上涨占比: {(df_result['ca_pct'] > 0).mean()*100:.1f}%")

    print(f"\n集合竞价成交额(万):")
    print(f"  均值: {df_result['amount_ca'].mean()/10000:.2f}")
    print(f"  中位数: {df_result['amount_ca'].median()/10000:.2f}")
