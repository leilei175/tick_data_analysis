"""
下载Tushare每日行情数据 (daily) 和每日基本面数据 (daily_basic)
保存到 daily_data 目录
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

import tushare as ts
import pandas as pd

# 读取配置文件
config_path = Path(__file__).parent / "config.py"
exec(config_path.read_text())

TOKEN = tushare_tk
print(f"Tushare Token: {TOKEN[:10]}...")

# 初始化Tushare
pro = ts.pro_api(TOKEN)

# 数据保存目录
DATA_DIR = Path(__file__).parent / "daily_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DAILY_DIR = DATA_DIR / "daily"
DAILY_BASIC_DIR = DATA_DIR / "daily_basic"
DAILY_DIR.mkdir(parents=True, exist_ok=True)
DAILY_BASIC_DIR.mkdir(parents=True, exist_ok=True)

# daily API 的字段
DAILY_FIELDS = ",".join([
    "ts_code", "trade_date", "open", "high", "low", "close",
    "pre_close", "change", "pct_chg", "vol", "amount"
])

# daily_basic API 的字段
DAILY_BASIC_FIELDS = ",".join([
    "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f",
    "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
    "dv_ratio", "dv_ttm", "total_share", "float_share",
    "free_share", "total_mv", "circ_mv"
])

def get_trading_days(start_date, end_date):
    """获取交易日列表"""
    try:
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        trading_days = df[df['is_open'] == 1]['cal_date'].tolist()
        return trading_days
    except Exception as e:
        print(f"获取交易日失败: {e}")
        return []

def download_daily(trade_date):
    """下载单日daily数据"""
    try:
        df = pro.daily(trade_date=trade_date, fields=DAILY_FIELDS)
        if df is not None and len(df) > 0:
            return df
        return None
    except Exception as e:
        print(f"下载 {trade_date} daily数据失败: {e}")
        return None

def download_daily_basic(trade_date):
    """下载单日daily_basic数据"""
    try:
        df = pro.daily_basic(trade_date=trade_date, fields=DAILY_BASIC_FIELDS)
        if df is not None and len(df) > 0:
            return df
        return None
    except Exception as e:
        print(f"下载 {trade_date} daily_basic数据失败: {e}")
        return None

def download_daily_range(start_date, end_date):
    """下载日期范围内的daily数据"""
    print(f"\n{'='*60}")
    print(f"下载每日行情数据 (daily)")
    print(f"日期范围: {start_date} - {end_date}")
    print(f"保存目录: {DAILY_DIR}")
    print(f"{'='*60}")

    trading_days = get_trading_days(start_date, end_date)
    print(f"\n交易日数量: {len(trading_days)}")

    all_data = []
    success_count = 0
    fail_count = 0

    for i, trade_date in enumerate(trading_days):
        print(f"\r处理中: {i+1}/{len(trading_days)} {trade_date}", end="")

        df = download_daily(trade_date)
        if df is not None and len(df) > 0:
            all_data.append(df)
            # 保存单日数据
            date_file = DAILY_DIR / f"daily_{trade_date}.parquet"
            df.to_parquet(date_file, index=False)
            success_count += 1

        time.sleep(0.1)  # API请求间隔

    print(f"\n\n完成! 成功: {success_count} 天, 失败: {fail_count} 天")

    # 保存完整数据
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_df = all_df.sort_values(['trade_date', 'ts_code'])

        output_file = DAILY_DIR / f"daily_{start_date}_{end_date}.parquet"
        all_df.to_parquet(output_file, index=False)
        print(f"\n完整数据已保存: {output_file}")
        print(f"总记录数: {len(all_df):,}")
        print(f"股票数量: {all_df['ts_code'].nunique():,}")
        print(f"日期范围: {start_date} - {end_date}")

    return all_df if all_data else None

def download_daily_basic_range(start_date, end_date):
    """下载日期范围内的daily_basic数据"""
    print(f"\n{'='*60}")
    print(f"下载每日基本面数据 (daily_basic)")
    print(f"日期范围: {start_date} - {end_date}")
    print(f"保存目录: {DAILY_BASIC_DIR}")
    print(f"{'='*60}")

    trading_days = get_trading_days(start_date, end_date)
    print(f"\n交易日数量: {len(trading_days)}")

    all_data = []
    success_count = 0
    fail_count = 0

    for i, trade_date in enumerate(trading_days):
        print(f"\r处理中: {i+1}/{len(trading_days)} {trade_date}", end="")

        df = download_daily_basic(trade_date)
        if df is not None and len(df) > 0:
            all_data.append(df)
            # 保存单日数据
            date_file = DAILY_BASIC_DIR / f"daily_basic_{trade_date}.parquet"
            df.to_parquet(date_file, index=False)
            success_count += 1

        time.sleep(0.1)  # API请求间隔

    print(f"\n\n完成! 成功: {success_count} 天, 失败: {fail_count} 天")

    # 保存完整数据
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_df = all_df.sort_values(['trade_date', 'ts_code'])

        output_file = DAILY_BASIC_DIR / f"daily_basic_{start_date}_{end_date}.parquet"
        all_df.to_parquet(output_file, index=False)
        print(f"\n完整数据已保存: {output_file}")
        print(f"总记录数: {len(all_df):,}")
        print(f"股票数量: {all_df['ts_code'].nunique():,}")
        print(f"日期范围: {start_date} - {end_date}")

    return all_df if all_data else None


# ==================== 数据读取函数 ====================

def get_daily(sec_list: list = None, start_day: str = None, end_day: str = None,
              data_dir: str = None) -> pd.DataFrame:
    """
    获取每日行情数据

    Args:
        sec_list: 股票代码列表，为空则获取全部
        start_day: 开始日期，格式 'YYYYMMDD'
        end_day: 结束日期，格式 'YYYYMMDD'
        data_dir: 数据目录路径

    Returns:
        DataFrame，包含列: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount

    Example:
        >>> df = get_daily(['600000.SH', '000001.SZ'], '20260101', '20260110')
    """
    if data_dir is None:
        data_dir = DAILY_DIR
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_files = sorted(data_dir.glob("daily_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件")

    start_date = int(start_day) if start_day else 0
    end_date = int(end_day) if end_day else 99991231

    filtered_files = []
    for f in all_files:
        fname = f.name.replace("daily_", "").replace(".parquet", "")
        if "_" in fname:
            parts = fname.split("_")
            file_start = int(parts[0])
            file_end = int(parts[1])
            if file_end >= start_date and file_start <= end_date:
                filtered_files.append(f)
        else:
            file_date = int(fname)
            if start_date <= file_date <= end_date:
                filtered_files.append(f)

    print(f"加载 {len(filtered_files)} 个数据文件...")

    all_dfs = []
    for f in filtered_files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"读取 {f.name} 失败: {e}")

    if not all_dfs:
        raise ValueError("无法读取任何数据")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['trade_date'] = combined_df['trade_date'].astype(int)
    combined_df = combined_df[
        (combined_df['trade_date'] >= start_date) &
        (combined_df['trade_date'] <= end_date)
    ]

    if sec_list:
        combined_df = combined_df[combined_df['ts_code'].isin(sec_list)]

    cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
            'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    result_df = combined_df[[c for c in cols if c in combined_df.columns]].copy()
    result_df = result_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
    result_df = result_df.sort_values(['ts_code', 'trade_date'])
    result_df['trade_date'] = result_df['trade_date'].astype(str)

    print(f"返回 {len(result_df):,} 条记录，{result_df['ts_code'].nunique()} 只股票")

    return result_df


def get_daily_basic(indicator: str = None, sec_list: list = None, start_day: str = None,
                   end_day: str = None, data_dir: str = None) -> pd.DataFrame:
    """
    获取每日基本面数据

    Args:
        indicator: 指标名称，为空则返回所有字段
        sec_list: 股票代码列表，为空则获取全部
        start_day: 开始日期，格式 'YYYYMMDD'
        end_day: 结束日期，格式 'YYYYMMDD'
        data_dir: 数据目录路径

    Returns:
        DataFrame

    Example:
        >>> df = get_daily_basic('turnover_rate', ['600000.SH'], '20260101', '20260110')
    """
    valid_indicators = [
        'close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
        'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
        'dv_ratio', 'dv_ttm', 'total_share', 'float_share',
        'free_share', 'total_mv', 'circ_mv'
    ]

    if data_dir is None:
        data_dir = DAILY_BASIC_DIR
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_files = sorted(data_dir.glob("daily_basic_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件")

    start_date = int(start_day) if start_day else 0
    end_date = int(end_day) if end_day else 99991231

    filtered_files = []
    for f in all_files:
        fname = f.name.replace("daily_basic_", "").replace(".parquet", "")
        if "_" in fname:
            parts = fname.split("_")
            file_start = int(parts[0])
            file_end = int(parts[1])
            if file_end >= start_date and file_start <= end_date:
                filtered_files.append(f)
        else:
            file_date = int(fname)
            if start_date <= file_date <= end_date:
                filtered_files.append(f)

    print(f"加载 {len(filtered_files)} 个数据文件...")

    all_dfs = []
    for f in filtered_files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"读取 {f.name} 失败: {e}")

    if not all_dfs:
        raise ValueError("无法读取任何数据")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['trade_date'] = combined_df['trade_date'].astype(int)
    combined_df = combined_df[
        (combined_df['trade_date'] >= start_date) &
        (combined_df['trade_date'] <= end_date)
    ]

    if sec_list:
        combined_df = combined_df[combined_df['ts_code'].isin(sec_list)]

    cols = ['ts_code', 'trade_date']
    if indicator:
        if indicator not in valid_indicators:
            raise ValueError(f"无效的指标: {indicator}")
        cols.append(indicator)
    else:
        cols.extend(valid_indicators)

    result_df = combined_df[[c for c in cols if c in combined_df.columns]].copy()
    result_df = result_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
    result_df = result_df.sort_values(['ts_code', 'trade_date'])
    result_df['trade_date'] = result_df['trade_date'].astype(str)

    print(f"返回 {len(result_df):,} 条记录，{result_df['ts_code'].nunique()} 只股票")

    return result_df


if __name__ == "__main__":
    # 默认日期范围
    start_date = '20250101'
    end_date = datetime.now().strftime('%Y%m%d')

    if len(sys.argv) > 1:
        end_date = sys.argv[1]
    if len(sys.argv) > 2:
        start_date = sys.argv[2]

    print(f"\n准备下载数据...")
    print(f"日期范围: {start_date} - {end_date}")

    # 下载daily数据
    download_daily_range(start_date, end_date)

    # 下载daily_basic数据
    download_daily_basic_range(start_date, end_date)

    print(f"\n{'='*60}")
    print("全部数据下载完成!")
    print(f"数据保存位置: {DATA_DIR}")
    print(f"  - daily: {DAILY_DIR}")
    print(f"  - daily_basic: {DAILY_BASIC_DIR}")
    print(f"{'='*60}")
