"""
下载Tushare每日基本面数据 (daily_basic)
保存到 daily_data/daily_basic 目录
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
DATA_DIR = Path(__file__).parent / "daily_data" / "daily_basic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 需要获取的字段
FIELDS = ",".join([
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

def download_daily_basic(trade_date, fields=FIELDS):
    """下载单日daily_basic数据"""
    try:
        # Tushare daily_basic单次最大返回5000条记录
        df = pro.daily_basic(trade_date=trade_date, fields=fields)
        if df is not None and len(df) > 0:
            return df
        return None
    except Exception as e:
        print(f"下载 {trade_date} 数据失败: {e}")
        return None

def download_date_range(start_date, end_date):
    """下载日期范围内的所有daily_basic数据"""
    print(f"\n{'='*60}")
    print(f"下载每日基本面数据")
    print(f"日期范围: {start_date} - {end_date}")
    print(f"保存目录: {DATA_DIR}")
    print(f"{'='*60}")

    # 获取交易日列表
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
            date_file = DATA_DIR / f"daily_basic_{trade_date}.parquet"
            df.to_parquet(date_file, index=False)
            success_count += 1
            print(f" -> {len(df):,} 条记录")
        else:
            fail_count += 1

        # API请求间隔，避免超过频率限制
        time.sleep(0.1)

    print(f"\n\n完成! 成功: {success_count} 天, 失败: {fail_count} 天")

    # 保存完整数据
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_df = all_df.sort_values(['trade_date', 'ts_code'])

        # 按日期分块保存，避免文件过大
        unique_dates = all_df['trade_date'].unique()
        start_d = unique_dates[0]
        end_d = unique_dates[-1]

        output_file = DATA_DIR / f"daily_basic_{start_d}_{end_d}.parquet"
        all_df.to_parquet(output_file, index=False)
        print(f"\n完整数据已保存: {output_file}")
        print(f"总记录数: {len(all_df):,}")
        print(f"股票数量: {all_df['ts_code'].nunique():,}")
        print(f"日期范围: {start_d} - {end_d}")

    return all_df if all_data else None

def download_latest_data(days=5):
    """下载最近N个交易日的数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days*7)).strftime('%Y%m%d')  # 预留足够时间

    return download_date_range(start_date, end_date)

# ==================== 数据读取函数 ====================

def get_data(indicator: str, sec_list: list, start_day: str, end_day: str,
             data_dir: str = None) -> pd.DataFrame:
    """
    获取指定股票列表、指标和日期范围的数据

    Args:
        indicator: 指标名称，支持:
            - 'close': 收盘价
            - 'turnover_rate': 换手率(%)
            - 'turnover_rate_f': 换手率(流通股)(%)
            - 'volume_ratio': 量比
            - 'pe': 市盈率
            - 'pe_ttm': 市盈率TTM
            - 'pb': 市净率
            - 'ps': 市销率
            - 'ps_ttm': 市销率TTM
            - 'dv_ratio': 股息率(%)
            - 'dv_ttm': 股息率TTM(%)
            - 'total_share': 总股本(万股)
            - 'float_share': 流通股本(万股)
            - 'free_share': 限售股本(万股)
            - 'total_mv': 总市值(万元)
            - 'circ_mv': 流通市值(万元)
        sec_list: 股票代码列表，如 ['600000.SH', '000001.SZ']
        start_day: 开始日期，格式 'YYYYMMDD'
        end_day: 结束日期，格式 'YYYYMMDD'
        data_dir: 数据目录路径，默认使用当前配置的目录

    Returns:
        DataFrame，包含列: ts_code, trade_date, {indicator}

    Example:
        >>> df = get_data('turnover_rate', ['600000.SH', '000001.SZ'], '20260101', '20260110')
        >>> print(df.head())
    """
    # 验证指标是否有效
    valid_indicators = [
        'close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
        'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
        'dv_ratio', 'dv_ttm', 'total_share', 'float_share',
        'free_share', 'total_mv', 'circ_mv'
    ]

    if indicator not in valid_indicators:
        raise ValueError(f"无效的指标: {indicator}，支持的指标: {valid_indicators}")

    # 设置数据目录
    if data_dir is None:
        data_dir = Path(__file__).parent / "daily_data" / "daily_basic"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # 获取所有数据文件
    all_files = sorted(data_dir.glob("daily_basic_*.parquet"))

    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件")

    # 确定日期范围
    start_date = int(start_day)
    end_date = int(end_day)

    # 筛选日期范围内的文件
    filtered_files = []
    for f in all_files:
        # 文件名格式: daily_basic_YYYYMMDD.parquet 或 daily_basic_YYYYMMDD_YYYYMMDD.parquet
        fname = f.name.replace("daily_basic_", "").replace(".parquet", "")

        if "_" in fname:
            # 合并文件格式: start_end
            parts = fname.split("_")
            file_start = int(parts[0])
            file_end = int(parts[1])
            if file_end >= start_date and file_start <= end_date:
                filtered_files.append(f)
        else:
            # 单日文件格式: YYYYMMDD
            file_date = int(fname)
            if start_date <= file_date <= end_date:
                filtered_files.append(f)

    if not filtered_files:
        raise ValueError(f"在日期范围 {start_day}-{end_day} 内未找到数据文件")

    print(f"加载 {len(filtered_files)} 个数据文件...")

    # 读取并合并数据
    all_dfs = []
    for f in filtered_files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"读取 {f.name} 失败: {e}")

    if not all_dfs:
        raise ValueError("无法读取任何数据")

    # 合并所有数据
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 筛选日期范围
    combined_df['trade_date'] = combined_df['trade_date'].astype(int)
    combined_df = combined_df[
        (combined_df['trade_date'] >= start_date) &
        (combined_df['trade_date'] <= end_date)
    ]

    # 筛选股票列表
    if sec_list:
        combined_df = combined_df[combined_df['ts_code'].isin(sec_list)]

    # 选择需要的列并去重
    cols = ['ts_code', 'trade_date', indicator]
    result_df = combined_df[cols].copy()

    # 去重（保留第一条记录）
    result_df = result_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')

    # 排序
    result_df = result_df.sort_values(['ts_code', 'trade_date'])
    result_df['trade_date'] = result_df['trade_date'].astype(str)

    print(f"返回 {len(result_df):,} 条记录，{result_df['ts_code'].nunique()} 只股票")

    return result_df


def get_turnover_rate(sec_list: list, start_day: str, end_day: str) -> pd.DataFrame:
    """便捷函数：获取换手率数据"""
    return get_data('turnover_rate', sec_list, start_day, end_day)


def get_close(sec_list: list, start_day: str, end_day: str) -> pd.DataFrame:
    """便捷函数：获取收盘价数据"""
    return get_data('close', sec_list, start_day, end_day)


def get_pe(sec_list: list, start_day: str, end_day: str) -> pd.DataFrame:
    """便捷函数：获取市盈率数据"""
    return get_data('pe', sec_list, start_day, end_day)


def get_pb(sec_list: list, start_day: str, end_day: str) -> pd.DataFrame:
    """便捷函数：获取市净率数据"""
    return get_data('pb', sec_list, start_day, end_day)


def get_market_cap(sec_list: list, start_day: str, end_day: str) -> pd.DataFrame:
    """便捷函数：获取总市值数据"""
    return get_data('total_mv', sec_list, start_day, end_day)


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 默认下载最近5天的数据
    days = 5
    if len(sys.argv) > 1:
        days = int(sys.argv[1])

    print(f"准备下载最近 {days} 个交易日的每日基本面数据...")

    # 检查API积分
    try:
        user_info = pro.query('fut_daily')  # 测试API连接
    except Exception as e:
        print(f"API连接测试: {e}")

    # 下载数据
    download_latest_data(days)
