"""
数据下载适配器模块
提供向后兼容的API，实际调用tushare_downloader.py中的函数
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入实际的下载函数
from tushare_downloader import download_daily as _download_daily
from tushare_downloader import download_daily_basic as _download_daily_basic
from financial_downloader import download_income as _download_income
from financial_downloader import download_balance as _download_balance
from financial_downloader import download_cashflow as _download_cashflow

# 数据目录
DATA_DIR = Path('./daily_data')


def get_daily(start_day: str, end_day: str) -> pd.DataFrame:
    """
    获取日线数据（向后兼容）
    
    Args:
        start_day: 开始日期 YYYYMMDD
        end_day: 结束日期 YYYYMMDD
        
    Returns:
        DataFrame: 日线数据
    """
    # 检查是否已有本地数据
    daily_dir = DATA_DIR / 'daily'
    
    # 如果本地有数据，直接读取
    if daily_dir.exists():
        files = list(daily_dir.glob(f'daily_{start_day}_{end_day}.parquet'))
        if files:
            return pd.read_parquet(files[0])
    
    # 否则下载
    return _download_daily(start_day, end_day)


def get_close(sec_list: Optional[List[str]], start_day: str, end_day: str) -> pd.DataFrame:
    """
    获取收盘价数据（向后兼容）
    
    Args:
        sec_list: 股票列表（未使用，保持兼容）
        start_day: 开始日期 YYYYMMDD
        end_day: 结束日期 YYYYMMDD
        
    Returns:
        DataFrame: 收盘价数据
    """
    daily_basic_dir = DATA_DIR / 'daily_basic'
    
    # 如果本地有数据，直接读取
    if daily_basic_dir.exists():
        files = list(daily_basic_dir.glob(f'daily_basic_{start_day}_{end_day}.parquet'))
        if files:
            df = pd.read_parquet(files[0])
            if sec_list:
                df = df[df['ts_code'].isin(sec_list)]
            return df[['ts_code', 'trade_date', 'close']]
    
    # 否则下载
    df = _download_daily_basic(start_day, end_day)
    if sec_list:
        df = df[df['ts_code'].isin(sec_list)]
    return df[['ts_code', 'trade_date', 'close']]


def get_data(indicator: str, sec_list: List[str], start_day: str, end_day: str,
             data_dir: str = None) -> pd.DataFrame:
    """
    获取指定指标数据（向后兼容）
    
    Args:
        indicator: 指标名称
        sec_list: 股票列表
        start_day: 开始日期
        end_day: 结束日期
        data_dir: 数据目录
        
    Returns:
        DataFrame: 指标数据
    """
    if indicator == 'close':
        return get_close(sec_list, start_day, end_day)
    
    daily_basic_dir = DATA_DIR / 'daily_basic' if data_dir is None else Path(data_dir)
    
    # 尝试从本地读取
    if daily_basic_dir.exists():
        files = list(daily_basic_dir.glob('*.parquet'))
        if files:
            dfs = []
            for f in files:
                df = pd.read_parquet(f)
                if indicator in df.columns:
                    dfs.append(df[['ts_code', 'trade_date', indicator]])
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                result = result[
                    (result['trade_date'] >= start_day) & 
                    (result['trade_date'] <= end_day)
                ]
                if sec_list:
                    result = result[result['ts_code'].isin(sec_list)]
                return result
    
    # 下载
    df = _download_daily_basic(start_day, end_day)
    if sec_list:
        df = df[df['ts_code'].isin(sec_list)]
    return df[['ts_code', 'trade_date', indicator]]


# 便捷函数
def get_turnover_rate(sec_list: List[str], start_day: str, end_day: str) -> pd.DataFrame:
    """获取换手率数据"""
    return get_data('turnover_rate', sec_list, start_day, end_day)


def get_pe(sec_list: List[str], start_day: str, end_day: str) -> pd.DataFrame:
    """获取市盈率数据"""
    return get_data('pe', sec_list, start_day, end_day)


def get_pb(sec_list: List[str], start_day: str, end_day: str) -> pd.DataFrame:
    """获取市净率数据"""
    return get_data('pb', sec_list, start_day, end_day)


def get_income(start_period: str, end_period: str) -> pd.DataFrame:
    """
    获取利润表数据
    
    Args:
        start_period: 开始报告期 YYYYMMDD
        end_period: 结束报告期 YYYYMMDD
        
    Returns:
        DataFrame: 利润表数据
    """
    income_dir = DATA_DIR / 'income'
    
    # 尝试从本地读取
    if income_dir.exists():
        files = list(income_dir.glob('income_all.parquet'))
        if files:
            df = pd.read_parquet(files[0])
            df = df[
                (df['end_date'] >= start_period) & 
                (df['end_date'] <= end_period)
            ]
            return df
    
    return _download_income(start_period, end_period)


def get_balance(start_period: str, end_period: str) -> pd.DataFrame:
    """
    获取资产负债表数据
    
    Args:
        start_period: 开始报告期 YYYYMMDD
        end_period: 结束报告期 YYYYMMDD
        
    Returns:
        DataFrame: 资产负债表数据
    """
    balance_dir = DATA_DIR / 'balance'
    
    # 尝试从本地读取
    if balance_dir.exists():
        files = list(balance_dir.glob('balance_all.parquet'))
        if files:
            df = pd.read_parquet(files[0])
            df = df[
                (df['end_date'] >= start_period) & 
                (df['end_date'] <= end_period)
            ]
            return df
    
    return _download_balance(start_period, end_period)


def get_cashflow(start_period: str, end_period: str) -> pd.DataFrame:
    """
    获取现金流量表数据
    
    Args:
        start_period: 开始报告期 YYYYMMDD
        end_period: 结束报告期 YYYYMMDD
        
    Returns:
        DataFrame: 现金流量表数据
    """
    cashflow_dir = DATA_DIR / 'cashflow'
    
    # 尝试从本地读取
    if cashflow_dir.exists():
        files = list(cashflow_dir.glob('cashflow_all.parquet'))
        if files:
            df = pd.read_parquet(files[0])
            df = df[
                (df['end_date'] >= start_period) & 
                (df['end_date'] <= end_period)
            ]
            return df
    
    return _download_cashflow(start_period, end_period)


# 交易日相关函数
def get_trading_days(start_date: str, end_date: str) -> list:
    """
    获取交易日列表
    
    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        
    Returns:
        list: 交易日列表
    """
    from mylib.tushare_client import get_trading_days
    return get_trading_days(start_date, end_date)
