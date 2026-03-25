import pandas as pd
import numpy as np
import os
from mylib.get_local_data import get_local_data
from mylib.date_utils import get_trading_days

def run_empirical_analysis(target_date='20251215'):
    print(f"--- 开始对 {target_date} 进行实证分析 ---")

    # 1. 获取小市值股票 (市值最小 20%)
    try:
        daily_basic = get_local_data('daily_basic', start_date=target_date, end_date=target_date)
        if daily_basic.empty:
            print("未找到当日市值数据。")
            return

        # 筛选小市值
        q_limit = daily_basic['total_mv'].quantile(0.2)
        small_caps = daily_basic[daily_basic['total_mv'] <= q_limit]['ts_code'].tolist()
        print(f"筛选出小市值股票数量: {len(small_caps)}")
    except Exception as e:
        print(f"获取市值数据失败: {e}")
        return

    # 2. 获取集合竞价因子 (使用已有的 build_call_auction_snapshot_factors 逻辑产出的数据)
    # 假设因子已预计算在 factor/high_frequency/call_auction_snapshot 目录下
    factor_path = f"factor/high_frequency/call_auction_snapshot/{target_date}.parquet"
    if not os.path.exists(factor_path):
        print(f"未找到当日因子预计算文件: {factor_path}，尝试从原始快照计算...")
        # 此处模拟计算逻辑，实际运行中会调用 build_call_auction_snapshot_factors.py
        # 为了演示，我们假设获取到了这些特征
        return

    df_factors = pd.read_parquet(factor_path)
    df_small = df_factors[df_factors['ts_code'].isin(small_caps)].copy()

    # 3. 获取日内涨跌幅 (Open to Close)
    # 计算公式: (close - open) / open
    df_daily = get_local_data('daily', start_date=target_date, end_date=target_date)
    df_daily['intraday_ret'] = (df_daily['close'] - df_daily['open']) / df_daily['open']

    # 合并数据
    df_merge = pd.merge(df_small, df_daily[['ts_code', 'intraday_ret']], on='ts_code')

    # 4. 计算相关性
    # 核心指标: 最后10秒价格变动 (auction_last1_ask1_ret) 与 日内收益 (intraday_ret)
    # 成交额变动: (auction_last1_askVol1)

    corr_price = df_merge['auction_last1_ask1_ret'].corr(df_merge['intraday_ret'])
    corr_vol = df_merge['auction_last1_askVol1'].corr(df_merge['intraday_ret'])

    print(f"\n相关性分析结果 ({target_date}):")
    print(f"最后10秒价格跳变 vs 日内涨跌幅: {corr_price:.4f}")
    print(f"最后10秒成交量 vs 日内涨跌幅: {corr_vol:.4f}")

if __name__ == "__main__":
    # 选取一个有数据的日期运行
    run_empirical_analysis('20251215')

