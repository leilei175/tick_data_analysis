"""
日期处理工具模块
提供统一的日期解析和转换功能
"""

from datetime import datetime, date
from typing import Union, Optional


def parse_date(date_input: Union[str, datetime, date]) -> datetime:
    """
    统一日期解析
    
    Args:
        date_input: 日期输入，支持字符串(YYYYMMDD或YYYY-MM-DD)、datetime、date
        
    Returns:
        datetime: datetime对象
        
    Examples:
        >>> parse_date('20250101')
        datetime.datetime(2025, 1, 1, 0, 0)
        >>> parse_date('2025-01-01')
        datetime.datetime(2025, 1, 1, 0, 0)
    """
    if isinstance(date_input, datetime):
        return date_input
    elif isinstance(date_input, date):
        return datetime.combine(date_input, datetime.min.time())
    elif isinstance(date_input, str):
        date_str = date_input.replace('-', '').replace('/', '')
        if len(date_str) == 8:
            return datetime.strptime(date_str, "%Y%m%d")
        else:
            raise ValueError(f"日期格式错误: {date_input}")
    else:
        raise ValueError(f"不支持的日期类型: {type(date_input)}")


def date_to_str(date_obj: Union[datetime, date, str], fmt: str = "%Y%m%d") -> str:
    """
    日期转字符串
    
    Args:
        date_obj: 日期对象或字符串
        fmt: 输出格式，默认"%Y%m%d"
        
    Returns:
        str: 格式化后的日期字符串
        
    Examples:
        >>> date_to_str(datetime(2025, 1, 1))
        '20250101'
        >>> date_to_str('2025-01-01')
        '20250101'
    """
    if isinstance(date_obj, str):
        # 如果是字符串，先解析再格式化
        return parse_date(date_obj).strftime(fmt)
    elif isinstance(date_obj, datetime):
        return date_obj.strftime(fmt)
    elif isinstance(date_obj, date):
        return date_obj.strftime(fmt)
    else:
        raise ValueError(f"不支持的日期类型: {type(date_obj)}")


def get_quarter_from_date(date_input: Union[str, datetime, date]) -> str:
    """
    从日期获取季度结束日期
    
    Args:
        date_input: 日期输入
        
    Returns:
        str: 季度结束日期，格式 YYYYMMDD
        
    Examples:
        >>> get_quarter_from_date('20250115')
        '20250331'
        >>> get_quarter_from_date('20250620')
        '20250630'
    """
    dt = parse_date(date_input)
    year = dt.year
    month = dt.month
    
    if month <= 3:
        return f"{year}0331"
    elif month <= 6:
        return f"{year}0630"
    elif month <= 9:
        return f"{year}0930"
    else:
        return f"{year}1231"


def get_date_list(start_date: str, end_date: str, freq: str = 'Q') -> list:
    """
    获取日期列表
    
    Args:
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        freq: 频率，'D'=日, 'M'=月, 'Q'=季度，默认'Q'
        
    Returns:
        list: 日期列表，格式 YYYYMMDD
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    if freq == 'D':
        # 日频
        dates = pd.date_range(start=start, end=end, freq='D')
    elif freq == 'M':
        # 月频
        dates = pd.date_range(start=start, end=end, freq='M')
    elif freq == 'Q':
        # 季频
        dates = pd.date_range(start=start, end=end, freq='Q')
    else:
        raise ValueError(f"不支持的频率: {freq}")
    
    return [d.strftime("%Y%m%d") for d in dates]


def get_month_dates(year: int, month: int) -> tuple:
    """
    获取某年某月的开始和结束日期
    
    Args:
        year: 年份
        month: 月份
        
    Returns:
        tuple: (start_date, end_date)，格式 YYYYMMDD
    """
    import calendar
    
    start_date = f"{year}{month:02d}01"
    _, last_day = calendar.monthrange(year, month)
    end_date = f"{year}{month:02d}{last_day:02d}"
    
    return start_date, end_date


def add_days(date_input: Union[str, datetime, date], days: int) -> str:
    """
    日期加减天数
    
    Args:
        date_input: 日期输入
        days: 天数（正数=加，负数=减）
        
    Returns:
        str: 结果日期，格式 YYYYMMDD
    """
    dt = parse_date(date_input)
    from datetime import timedelta
    result = dt + timedelta(days=days)
    return result.strftime("%Y%m%d")


# 导入pandas用于get_date_list函数
try:
    import pandas as pd
except ImportError:
    pd = None
