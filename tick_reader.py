"""
Tick数据读取框架
支持读取 tick_2026 目录下的 parquet 格式 tick 数据

目录结构:
    tick_2026/2026/YYYY/MM/DD/stock_code.parquet
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, date


class TickDataReader:
    """Tick数据读取器"""

    def __init__(self, base_path: str = "./tick_2026"):
        """
        初始化读取器

        Args:
            base_path: tick数据根目录
        """
        self.base_path = Path(base_path)
        self._schema_cache = None

    @property
    def schema(self) -> Optional[Any]:
        """获取数据模式"""
        if self._schema_cache is None:
            # 查找一个示例文件获取模式
            for f in self.base_path.rglob("*.parquet"):
                self._schema_cache = pq.read_schema(f)
                break
        return self._schema_cache

    def get_available_dates(self, year: str = "2026") -> List[date]:
        """获取可用日期列表"""
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
        return dates

    def get_available_stocks(self, date: Optional[date] = None) -> List[str]:
        """
        获取指定日期的股票列表

        Args:
            date: 指定日期，为None则返回所有股票

        Returns:
            股票代码列表
        """
        stocks = set()
        if date is None:
            for f in self.base_path.rglob("*.parquet"):
                stocks.add(f.stem)
        else:
            pattern = f"{date.year:04d}/{date.month:02d}/{date.day:02d}"
            for f in self.base_path.glob(f"{pattern}/*.parquet"):
                stocks.add(f.stem)
        return sorted(stocks)

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

        Returns:
            包含tick数据的DataFrame
        """
        dfs = []
        pattern = f"{self.base_path}/2026/*/*/{stock_code}.parquet"

        for f in Path(".").glob(pattern) if not Path(pattern).is_absolute() else []:
            pass

        # 使用正确的路径模式
        for f in self.base_path.glob(f"2026/*/*/{stock_code}.parquet"):
            file_date = self._get_date_from_path(f)
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            df = pd.read_parquet(f)
            df['file_date'] = file_date
            df['file_path'] = str(f)
            dfs.append(df)

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result = self._parse_time_column(result)
            return result.sort_values('time').reset_index(drop=True)
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

        Returns:
            包含tick数据的DataFrame
        """
        date_path = f"{target_date.year:04d}/{target_date.month:02d}/{target_date.day:02d}"
        dfs = []

        if stock_codes is None:
            pattern = f"{date_path}/*.parquet"
            for f in self.base_path.glob(pattern):
                df = pd.read_parquet(f)
                df['stock_code'] = f.stem
                dfs.append(df)
        else:
            for code in stock_codes:
                f = self.base_path / date_path / f"{code}.parquet"
                if f.exists():
                    df = pd.read_parquet(f)
                    df['stock_code'] = code
                    dfs.append(df)

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result = self._parse_time_column(result)
            return result.sort_values(['stock_code', 'time']).reset_index(drop=True)
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

        Returns:
            字典，key为股票代码，value为DataFrame
        """
        return {code: self.read_stock(code, start_date, end_date) for code in stock_codes}

    def get_column_info(self) -> Dict[str, str]:
        """获取字段说明"""
        return {
            'time': '时间戳(纳秒)',
            'lastPrice': '最新价',
            'open': '开盘价',
            'high': '最高价',
            'low': '最低价',
            'lastClose': '昨收价',
            'amount': '成交额',
            'volume': '成交量',
            'pvolume': '流通成交量',
            'tickvol': 'Tick成交量',
            'stockStatus': '股票状态',
            'openInt': '持仓量/流通股数',
            'lastSettlementPrice': '昨结算价',
            'askPrice': '卖盘价(5档)',
            'bidPrice': '买盘价(5档)',
            'askVol': '卖盘量(5档)',
            'bidVol': '买盘量(5档)',
            'settlementPrice': '结算价',
            'transactionNum': '成交笔数',
            'pe': '市盈率',
        }

    def _parse_time_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析时间列"""
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='ns')
            df['time_str'] = df['datetime'].dt.strftime('%H:%M:%S.%f')[:-3]
        return df

    def _get_date_from_path(self, path: Path) -> date:
        """从文件路径提取日期"""
        parts = path.parts
        # 路径格式: base/year/month/day/stock.parquet
        # 如: ./tick_2026/2026/01/05/000001.SZ.parquet
        year_idx = -4
        month_idx = -3
        day_idx = -2
        return date(int(parts[year_idx]), int(parts[month_idx]), int(parts[day_idx]))


class TickDataAnalyzer:
    """Tick数据分析器"""

    def __init__(self, reader: Optional[TickDataReader] = None):
        self.reader = reader or TickDataReader()

    def analyze_stock(self, stock_code: str) -> Dict[str, Any]:
        """分析单只股票"""
        df = self.reader.read_stock(stock_code)
        if df.empty:
            return {}

        return {
            'stock_code': stock_code,
            'total_records': len(df),
            'date_range': (df['file_date'].min(), df['file_date'].max()),
            'price_stats': {
                'open': df['open'].iloc[0] if not df['open'].isna().all() else None,
                'high': df['high'].max() if not df['high'].isna().all() else None,
                'low': df['low'].min() if not df['low'].isna().all() else None,
                'close': df['lastPrice'].iloc[-1] if not df['lastPrice'].isna().all() else None,
            },
            'volume_stats': {
                'total_volume': df['volume'].sum(),
                'avg_price': df['amount'].sum() / df['volume'].replace(0, float('nan')).sum() if df['volume'].sum() > 0 else None,
            },
        }

    def batch_analyze(self, stock_codes: List[str]) -> pd.DataFrame:
        """批量分析多只股票"""
        results = []
        for code in stock_codes:
            stats = self.analyze_stock(code)
            if stats:
                results.append(stats)
        return pd.DataFrame(results) if results else pd.DataFrame()


if __name__ == "__main__":
    # 示例用法
    reader = TickDataReader("./tick_2026")

    # 获取可用日期
    dates = reader.get_available_dates("2026")
    print(f"可用日期数量: {len(dates)}")
    print(f"前5个日期: {dates[:5]}")

    # 获取可用股票
    stocks = reader.get_available_stocks(dates[0]) if dates else []
    print(f"\n{dates[0] if dates else 'N/A'} 的股票数量: {len(stocks)}")
    print(f"前10只股票: {stocks[:10]}")

    # 读取单只股票示例
    if stocks:
        df = reader.read_stock(stocks[0])
        print(f"\n读取 {stocks[0]} 数据:")
        print(f"记录数: {len(df)}")
        print(f"列: {df.columns.tolist()}")
        print(df.head(3))

    # 分析示例
    print("\n" + "="*50)
    analyzer = TickDataAnalyzer(reader)
    if stocks:
        stats = analyzer.analyze_stock(stocks[0])
        print(f"\n{stocks[0]} 分析结果:")
        print(stats)
