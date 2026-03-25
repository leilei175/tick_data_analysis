"""
Tick数据获取模块
根据给定的股票代码和日期范围获取tick数据
"""

from pathlib import Path
from datetime import date, timedelta
from typing import Union, List, Optional, Dict, Iterable, Tuple
import re
import numpy as np
import pandas as pd


# 默认tick数据目录
DEFAULT_TICK_DIR = './tick_2026/'
DEFAULT_TICK_TIMEZONE = 'Asia/Shanghai'


class TickDataReader:
    """Tick数据读取器"""

    def __init__(self, base_path: str = DEFAULT_TICK_DIR):
        """
        初始化读取器

        Args:
            base_path: tick数据根目录
        """
        self.base_path = Path(base_path)
        self._available_dates_cache: Dict[str, List[date]] = {}
        self._date_stock_cache: Dict[date, List[str]] = {}
        self._stock_files_cache: Dict[str, List[Tuple[date, Path]]] = {}

    def get_available_dates(self, year: str = "2026") -> List[date]:
        """获取可用日期列表"""
        cached = self._available_dates_cache.get(year)
        if cached is not None:
            return list(cached)

        dates = []
        year_path = self.base_path / year
        if not year_path.exists():
            return dates

        for month_dir in sorted(year_path.iterdir()):
            if month_dir.is_dir():
                for day_dir in sorted(month_dir.iterdir()):
                    if day_dir.is_dir():
                        try:
                            d = date(int(year), int(month_dir.name), int(day_dir.name))
                            dates.append(d)
                        except ValueError:
                            continue
        self._available_dates_cache[year] = dates
        return dates

    def get_available_stocks(self, dt: Optional[date] = None) -> List[str]:
        """
        获取指定日期的股票列表

        Args:
            dt: 指定日期，为None则返回所有股票
        """
        if dt is None:
            stocks = set()
            for f in self.base_path.rglob("*.parquet"):
                stocks.add(f.stem)
            return sorted(stocks)

        cached = self._date_stock_cache.get(dt)
        if cached is not None:
            return list(cached)

        stocks = sorted(f.stem for f in self._iter_date_files(dt))
        self._date_stock_cache[dt] = stocks
        return list(stocks)

    def read_stock(
        self,
        stock_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        读取单只股票的tick数据

        Args:
            stock_code: 股票代码，如 "300044.SZ"
            start_date: 开始日期
            end_date: 结束日期
        """
        dfs = []
        stock_code = _normalize_stock_code(stock_code)
        stock_files = self._get_stock_files(stock_code, start_date, end_date)

        for _, file_path in stock_files:
            df = pd.read_parquet(file_path)
            dfs.append(df)

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result = self._parse_time_column(result)
            return self._finalize_dataframe(result)
        return pd.DataFrame()

    def read_date(
        self,
        target_date: date,
        stock_codes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        读取指定日期的所有或指定股票的tick数据

        Args:
            target_date: 目标日期
            stock_codes: 股票代码列表，为None则读取所有
        """
        date_path = f"{target_date.year:04d}/{target_date.month:02d}/{target_date.day:02d}"
        dfs = []

        if stock_codes is None:
            for f in self._iter_date_files(target_date):
                df = pd.read_parquet(f)
                df['stock_code'] = f.stem
                dfs.append(df)
        else:
            for code in (_normalize_stock_code(code) for code in stock_codes):
                f = self.base_path / date_path / f"{code}.parquet"
                if f.exists():
                    df = pd.read_parquet(f)
                    df['stock_code'] = code
                    dfs.append(df)

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result = self._parse_time_column(result)
            return self._finalize_dataframe(result, sort_by_stock='stock_code' in result.columns)
        return pd.DataFrame()

    def read_multiple_stocks(
        self,
        stock_codes: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        读取多只股票的tick数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        """
        normalized_codes = [_normalize_stock_code(code) for code in stock_codes]
        if start_date and end_date and start_date == end_date:
            day_df = self.read_date(start_date, normalized_codes)
            return {
                code: day_df[day_df['stock_code'] == code].copy()
                for code in normalized_codes
            }
        return {code: self.read_stock(code, start_date, end_date) for code in normalized_codes}

    def _parse_time_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析时间列"""
        if 'time' in df.columns:
            dt_index = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TICK_TIMEZONE)
            df['_sort_datetime'] = dt_index
            df['day'] = dt_index.dt.strftime('%Y-%m-%d')
            df['time_ms'] = df['time']
            df['time'] = dt_index.dt.strftime('%H:%M:%S.%f').str[:-3]
        return df

    def _finalize_dataframe(self, df: pd.DataFrame, sort_by_stock: bool = False) -> pd.DataFrame:
        """统一输出格式：输出 day/time 列，并设置带时区的 DatetimeIndex。"""
        if df.empty:
            return df

        drop_cols = [col for col in ['file_date', 'file_path'] if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if 'day' not in df.columns and 'time' in df.columns:
            df = self._parse_time_column(df)

        if '_sort_datetime' in df.columns:
            if sort_by_stock and 'stock_code' in df.columns:
                df = df.sort_values(['stock_code', '_sort_datetime'])
            else:
                df = df.sort_values(['_sort_datetime'])
            df = df.set_index('_sort_datetime', drop=True)
            df.index.name = 'datetime'

        return df

    def _iter_date_files(self, dt: date) -> Iterable[Path]:
        """遍历指定日期目录下的 parquet 文件"""
        date_dir = self.base_path / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
        if not date_dir.exists():
            return []
        return sorted(date_dir.glob("*.parquet"))

    def _get_stock_files(
        self,
        stock_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Tuple[date, Path]]:
        """获取股票对应的 parquet 文件列表，按日期排序并缓存。"""
        cached = self._stock_files_cache.get(stock_code)
        if cached is None:
            pattern = f"{stock_code}.parquet"
            files = [
                (self._get_date_from_path(path), path)
                for path in self.base_path.rglob(pattern)
            ]
            files.sort(key=lambda item: item[0])
            self._stock_files_cache[stock_code] = files
            cached = files

        if start_date is None and end_date is None:
            return list(cached)

        if start_date is None:
            start_date = end_date
        if end_date is None:
            end_date = start_date
        if start_date is None or end_date is None:
            return []
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        return [
            (file_date, file_path)
            for file_date, file_path in cached
            if start_date <= file_date <= end_date
        ]

    def _get_stock_file(self, dt: date, stock_code: str) -> Path:
        """获取指定日期和股票对应的 parquet 文件路径"""
        return self.base_path / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}" / f"{stock_code}.parquet"

    def _resolve_target_dates(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> List[date]:
        """根据日期条件解析需要读取的交易日"""
        if start_date is None and end_date is None:
            years = sorted(p.name for p in self.base_path.iterdir() if p.is_dir() and p.name.isdigit()) if self.base_path.exists() else []
            dates: List[date] = []
            for year in years:
                dates.extend(self.get_available_dates(year))
            return dates

        if start_date is None:
            start_date = end_date
        if end_date is None:
            end_date = start_date
        if start_date is None or end_date is None:
            return []
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        candidate_dates = []
        current = start_date
        while current <= end_date:
            candidate_dates.append(current)
            current += timedelta(days=1)

        return [dt for dt in candidate_dates if self._date_exists(dt)]

    def _date_exists(self, dt: date) -> bool:
        """判断某个日期目录是否存在"""
        date_dir = self.base_path / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
        return date_dir.exists() and date_dir.is_dir()

    def _get_date_from_path(self, path: Path) -> date:
        """从文件路径提取日期"""
        parts = path.parts
        year_idx = -4
        month_idx = -3
        day_idx = -2
        return date(int(parts[year_idx]), int(parts[month_idx]), int(parts[day_idx]))


# 全局默认读取器
_default_reader = None


def get_tick_data(
    stock_codes: Union[str, List[str]],
    start_date: Union[str, date, None] = None,
    end_date: Union[str, date, None] = None,
    tick_dir: str = DEFAULT_TICK_DIR,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    获取tick数据的主要API接口

    Args:
        stock_codes: 股票代码或股票代码列表
                     - 单个股票: "000001.SZ" 或 "000001"
                     - 多个股票: ["000001.SZ", "300044.SZ"] 或 "000001.SZ,300044.SZ"
        start_date: 开始日期，格式: "YYYYMMDD" 或 "YYYY-MM-DD" 或 date对象
        end_date: 结束日期，格式: "YYYYMMDD" 或 "YYYY-MM-DD" 或 date对象，默认为start_date
        tick_dir: tick数据目录路径，默认为 "./tick_2026/"

    Returns:
        DataFrame或Dict:
        - 单个股票返回DataFrame
        - 多个股票返回 Dict[stock_code, DataFrame]

    Examples:
        # 读取单个股票全部数据
        df = get_tick_data("000001.SZ")

        # 读取单个股票指定日期
        df = get_tick_data("000001.SZ", start_date="20260101")

        # 读取单个股票日期范围
        df = get_tick_data("000001.SZ", start_date="20260101", end_date="20260110")

        # 读取多个股票
        result = get_tick_data(["000001.SZ", "300044.SZ"], start_date="20260101")

        # 使用字符串指定多个股票
        result = get_tick_data("000001.SZ,300044.SZ", start_date="20260101")
    """
    # 初始化读取器
    reader = _get_reader(tick_dir)

    # 解析股票代码
    if isinstance(stock_codes, str):
        if ',' in stock_codes:
            stock_codes = [s.strip() for s in stock_codes.split(',')]
        else:
            stock_codes = [_normalize_stock_code(stock_codes)]
    elif isinstance(stock_codes, (list, tuple)):
        stock_codes = [_normalize_stock_code(code) for code in stock_codes]

    # 解析日期
    start_dt = _parse_date(start_date) if start_date else None
    end_dt = _parse_date(end_date) if end_date else start_dt  # 默认等于start_date

    # 读取数据
    if len(stock_codes) == 1:
        # 单个股票
        return reader.read_stock(stock_codes[0], start_dt, end_dt)
    else:
        # 多个股票
        return reader.read_multiple_stocks(stock_codes, start_dt, end_dt)


def get_tick_data_short(
    stock_codes: Union[str, List[str]],
    start_date: Union[str, date, None] = None,
    end_date: Union[str, date, None] = None,
    tick_dir: str = DEFAULT_TICK_DIR,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    获取精简版 tick 数据，包含基础行情列和拆分后的五档盘口列。

    Returns:
        DataFrame 或 Dict[stock_code, DataFrame]
    """
    result = get_tick_data(stock_codes, start_date=start_date, end_date=end_date, tick_dir=tick_dir)

    if isinstance(result, dict):
        return {code: _shorten_tick_dataframe(df) for code, df in result.items()}
    return _shorten_tick_dataframe(result)


def _normalize_stock_code(code: str) -> str:
    """
    标准化股票代码

    Args:
        code: 原始股票代码

    Returns:
        标准化后的股票代码，如 "000001.SZ"
    """
    code = str(code).strip().upper()

    # 已经是标准格式时直接返回，避免把 118058.SH 误改成 118058.SZ
    if re.match(r'^\d{6}\.(SH|SZ)$', code):
        return code

    # 兼容 SH.600000 / SZ.000001 这种前缀写法
    if code.startswith('SH.'):
        return f"{code[3:]}.SH"
    if code.startswith('SZ.'):
        return f"{code[3:]}.SZ"

    # 如果没有后缀，根据代码自动添加
    if '.' not in code:
        if code.startswith(('110', '111', '113', '118')):
            return f"{code}.SH"
        if code.startswith(('123', '127', '128')):
            return f"{code}.SZ"
        if code.startswith('6'):
            return f"{code}.SH"
        return f"{code}.SZ"

    return code


def _get_reader(tick_dir: str) -> TickDataReader:
    """按 tick 目录复用读取器实例"""
    global _default_reader

    tick_dir_path = str(Path(tick_dir))
    if _default_reader is None:
        _default_reader = {}

    reader = _default_reader.get(tick_dir_path)
    if reader is None:
        reader = TickDataReader(tick_dir_path)
        _default_reader[tick_dir_path] = reader
    return reader


def _parse_date(dt: Union[str, date]) -> date:
    """
    解析日期字符串

    Args:
        dt: 日期字符串或date对象

    Returns:
        date对象
    """
    if isinstance(dt, date):
        return dt

    dt_str = str(dt).strip()

    # 移除常见分隔符
    dt_str = dt_str.replace('-', '').replace('/', '').replace('.', '')

    # 解析日期
    if len(dt_str) == 8:
        return date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
    else:
        raise ValueError(f"日期格式错误: {dt}，支持格式: YYYYMMDD, YYYY-MM-DD")


def _expand_level_column(df: pd.DataFrame, src_col: str, prefix: str) -> pd.DataFrame:
    """将五档数组列拆分为 prefix1..prefix5。"""
    if src_col not in df.columns:
        return pd.DataFrame(index=df.index)

    values = df[src_col].tolist()
    if not values:
        return pd.DataFrame(index=df.index)

    expanded = np.asarray(values)
    if expanded.ndim == 1:
        expanded = np.array([
            list(item) if isinstance(item, (list, tuple, np.ndarray)) else [item]
            for item in values
        ], dtype=object)

    if expanded.ndim != 2:
        return pd.DataFrame(index=df.index)

    max_levels = min(5, expanded.shape[1])
    columns = [f'{prefix}{i}' for i in range(1, max_levels + 1)]
    return pd.DataFrame(expanded[:, :max_levels], index=df.index, columns=columns)


def _shorten_tick_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """保留精简字段并拆分五档盘口。"""
    if df.empty:
        return df.copy()

    result = df.copy()
    base_cols = ['day', 'time', 'lastPrice', 'open', 'high', 'low', 'lastClose', 'amount', 'volume']
    if 'stock_code' in result.columns:
        base_cols.append('stock_code')

    parts = [result[[col for col in base_cols if col in result.columns]].copy()]
    parts.append(_expand_level_column(result, 'askPrice', 'askPrice'))
    parts.append(_expand_level_column(result, 'bidPrice', 'bidPrice'))
    parts.append(_expand_level_column(result, 'askVol', 'askVol'))
    parts.append(_expand_level_column(result, 'bidVol', 'bidVol'))

    short_df = pd.concat(parts, axis=1)
    short_df.index = result.index
    short_df.index.name = result.index.name
    short_df['diff_amount'] = short_df['amount'].diff()
    return short_df


def get_available_dates(year: str = "2026", tick_dir: str = DEFAULT_TICK_DIR) -> List[date]:
    """
    获取可用日期列表

    Args:
        year: 年份
        tick_dir: tick数据目录

    Returns:
        可用日期列表
    """
    reader = _get_reader(tick_dir)
    return reader.get_available_dates(year)


def get_available_stocks(
    dt: Optional[date] = None,
    tick_dir: str = DEFAULT_TICK_DIR
) -> List[str]:
    """
    获取指定日期的股票列表

    Args:
        dt: 日期
        tick_dir: tick数据目录

    Returns:
        股票代码列表
    """
    reader = _get_reader(tick_dir)
    return reader.get_available_stocks(dt)


# 示例用法
if __name__ == "__main__":
    # 示例1: 读取单个股票
    print("=" * 50)
    print("示例1: 读取单个股票")
    df = get_tick_data("000001.SZ")
    if not df.empty:
        print(f"记录数: {len(df)}")
        print(f"列: {df.columns.tolist()}")
        print(df.head(3))
    else:
        print("未找到数据")

    # 示例2: 读取指定日期
    print("\n" + "=" * 50)
    print("示例2: 读取指定日期")
    df = get_tick_data("000001.SZ", start_date="20260105")
    if not df.empty:
        print(f"记录数: {len(df)}")
        print(df.head(3))
    else:
        print("未找到数据")

    # 示例3: 读取日期范围
    print("\n" + "=" * 50)
    print("示例3: 读取日期范围")
    df = get_tick_data("000001.SZ", start_date="20260101", end_date="20260110")
    if not df.empty:
        print(f"记录数: {len(df)}")
        print(f"日期范围: {df.index.min()} ~ {df.index.max()}")
    else:
        print("未找到数据")

    # 示例4: 读取多个股票
    print("\n" + "=" * 50)
    print("示例4: 读取多个股票")
    result = get_tick_data(["000001.SZ", "300044.SZ"], start_date="20260105")
    for code, df in result.items():
        print(f"{code}: {len(df)} 条记录")
