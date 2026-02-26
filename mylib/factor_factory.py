"""
因子工厂模块
提供因子数据的统一访问接口，支持多种因子来源
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

# 添加父目录到路径
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


class FactorFactory:
    """
    因子工厂类

    提供因子的统一访问接口，支持以下因子来源：
    - high_frequency: 高频因子（来自factor/by_factor目录）
    - fundamental: 基本面因子（来自factor/fundamental目录）
    - technical: 技术指标因子（来自factor/technical目录）
    """

    # 因子来源配置
    SOURCES = {
        'high_frequency': {
            'dir': 'by_factor',
            'prefix': 'zz1000_',
            'description': '高频因子（盘口数据计算）'
        },
        'fundamental': {
            'dir': 'fundamental',
            'prefix': 'zz1000_',
            'description': '基本面因子（财务数据）'
        },
        'technical': {
            'dir': 'technical',
            'prefix': 'zz1000_',
            'description': '技术指标因子'
        }
    }

    # 预定义的因子列表
    PREDEFINED_FACTORS = {
        'high_frequency': [
            'bid_ask_spread',           # 买卖价差
            'vwap_deviation',          # VWAP偏离度
            'effective_spread',         # 有效价差
            'micro_price',              # 微观价格
            'depth_imbalance',          # 深度失衡
            'order_imbalance',          # 订单失衡
            'trade_imbalance',          # 交易失衡
            'trade_flow_intensity',     # 交易流强度
            'realized_volatility',      # 已实现波动率
            'price_momentum',           # 价格动量
        ],
        'fundamental': [
            'pe',                       # 市盈率
            'pb',                       # 市净率
            'ps',                       # 市销率
            'market_cap',               # 市值
            'roe',                      # 净资产收益率
            'profit_growth',            # 利润增长率
            'revenue_growth',           # 营收增长率
            'debt_to_assets',           # 资产负债率
            'current_ratio',            # 流动比率
        ],
        'technical': [
            'ma5',                      # 5日均线
            'ma10',                     # 10日均线
            'ma20',                     # 20日均线
            'ma60',                     # 60日均线
            'rsi',                      # 相对强弱指标
            'macd',                     # 移动平均收敛散度
            'boll_upper',               # 布林带上轨
            'boll_lower',               # 布林带下轨
            'atr',                      # 平均真实波幅
            'volume_ratio',             # 量比
        ]
    }

    STOCK_POOL_ALIASES = {
        'zz1000': 'zz1000',
        '000852': 'zz1000',
        'hs300': 'hs300',
        '000300': 'hs300',
        'zzxf': 'zzxf',
        'csi_consumer': 'zzxf',
        'zzcj': 'zzxf',
    }

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化因子工厂

        Args:
            base_dir: 因子数据根目录，默认为项目根目录下的factor目录
        """
        self.base_dir = Path(base_dir) if base_dir else Path(_parent_dir) / 'factor'
        self._cache = {}

    def get_base_dir(self) -> Path:
        """获取因子数据根目录"""
        return self.base_dir

    def list_sources(self) -> List[str]:
        """列出所有可用的因子来源"""
        return list(self.SOURCES.keys())

    def list_stock_pools(self) -> Dict[str, List[str]]:
        """
        列出所有可用的股票池

        Returns:
            Dict: {pool_name: [description, ...]}
        """
        return {
            'zz1000': ['中证1000成分股'],
            'hs300': ['沪深300成分股'],
            'zzxf': ['中证消费成分股'],
        }

    def _normalize_stock_pool(self, stock_pool: str) -> str:
        """标准化股票池标识"""
        if not stock_pool:
            return 'zz1000'
        return self.STOCK_POOL_ALIASES.get(stock_pool.lower(), stock_pool.lower())

    def list_factors(self, source: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
        """
        列出可用的因子

        Args:
            source: 因子来源，None表示返回所有来源的因子

        Returns:
            如果指定source，返回该来源的因子列表
            否则返回所有来源因子的字典
        """
        if source:
            if source not in self.SOURCES:
                raise ValueError(f"未知因子来源: {source}，可用来源: {list(self.SOURCES.keys())}")
            return self.PREDEFINED_FACTORS.get(source, [])

        # 返回所有因子
        all_factors = {}
        for src in self.SOURCES.keys():
            all_factors[src] = self.PREDEFINED_FACTORS.get(src, [])
        return all_factors

    def get_factor(
        self,
        factor_name: str,
        source: str = 'high_frequency',
        stock_pool: str = 'zz1000',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取因子数据

        Args:
            factor_name: 因子名称
            source: 因子来源
            stock_pool: 股票池
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD

        Returns:
            DataFrame: 宽格式因子数据，index为日期，columns为股票代码
        """
        normalized_pool = self._normalize_stock_pool(stock_pool)
        cache_key = f"{source}_{normalized_pool}_{factor_name}_{start_date}_{end_date}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # 获取因子数据目录
        source_config = self.SOURCES.get(source)
        if not source_config:
            raise ValueError(f"未知因子来源: {source}")

        factor_dir = self.base_dir / source_config['dir']

        # 尝试多种文件命名模式
        df = None

        candidate_pools = [normalized_pool]
        if normalized_pool != 'zz1000':
            candidate_pools.append('zz1000')

        loaded_pool = None
        for pool in candidate_pools:
            filepath = factor_dir / f"{pool}_{factor_name}.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                loaded_pool = pool
                break

            year_files = sorted(factor_dir.glob(f"{pool}_{factor_name}_*.parquet"))
            if year_files:
                dfs = [pd.read_parquet(f) for f in year_files]
                df = pd.concat(dfs, axis=0)
                loaded_pool = pool
                break

        if df is None:
            raise FileNotFoundError(
                f"无法找到因子文件: {normalized_pool}_{factor_name} in {factor_dir}"
            )

        if loaded_pool and loaded_pool != normalized_pool:
            print(f"警告: 未找到 {normalized_pool} 对应因子文件，回退使用 {loaded_pool}")

        # 统一索引为日期
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~df.index.isna()].sort_index()

        # 按日期过滤
        if start_date:
            start_ts = pd.to_datetime(start_date, format='%Y%m%d', errors='coerce')
            df = df[df.index >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date, format='%Y%m%d', errors='coerce')
            df = df[df.index <= end_ts]

        # 缓存
        self._cache[cache_key] = df.copy()

        return df.copy()

    def get_factor_info(self, factor_name: str, source: str = 'high_frequency') -> Dict:
        """
        获取因子信息

        Args:
            factor_name: 因子名称
            source: 因子来源

        Returns:
            Dict: 因子信息
        """
        try:
            df = self.get_factor(factor_name, source)

            flat_values = df.values.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]

            return {
                'factor': factor_name,
                'source': source,
                'shape': {'rows': len(df), 'cols': len(df.columns)},
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                },
                'stock_count': len(df.columns),
                'date_count': len(df),
                'statistics': {
                    'count': len(flat_values),
                    'mean': float(np.mean(flat_values)),
                    'std': float(np.std(flat_values)),
                    'min': float(np.min(flat_values)),
                    'max': float(np.max(flat_values)),
                    'median': float(np.median(flat_values)),
                    'skew': float(pd.Series(flat_values).skew()),
                    'kurtosis': float(pd.Series(flat_values).kurtosis())
                }
            }
        except Exception as e:
            return {
                'factor': factor_name,
                'source': source,
                'error': str(e)
            }

    def list_available_files(self, source: str) -> List[Dict]:
        """
        列出指定来源的可用因子文件

        Args:
            source: 因子来源

        Returns:
            List: 文件信息列表
        """
        source_config = self.SOURCES.get(source)
        if not source_config:
            raise ValueError(f"未知因子来源: {source}")

        factor_dir = self.base_dir / source_config['dir']

        if not factor_dir.exists():
            return []

        files = list(factor_dir.glob('*.parquet'))

        result = []
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            result.append({
                'filename': f.name,
                'file_size_mb': round(size_mb, 2),
                'file_path': str(f)
            })

        return result

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


# 全局因子工厂实例
_default_factory: Optional[FactorFactory] = None


def get_factory(base_dir: Optional[str] = None) -> FactorFactory:
    """
    获取全局因子工厂实例

    Args:
        base_dir: 因子数据根目录

    Returns:
        FactorFactory: 因子工厂实例
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = FactorFactory(base_dir)
    return _default_factory


def list_all_available_factors() -> Dict:
    """
    列出所有可用的因子信息（用于API返回）

    Returns:
        Dict: 包含来源、因子列表、股票池信息
    """
    factory = get_factory()
    return {
        'sources': factory.list_sources(),
        'factors': factory.list_factors(),
        'stock_pools': factory.list_stock_pools(),
        'data_directory': str(factory.get_base_dir())
    }


if __name__ == '__main__':
    # 测试代码
    factory = FactorFactory()

    print("可用因子来源:", factory.list_sources())
    print("\n可用因子:")
    print(factory.list_factors())

    print("\n可用股票池:")
    print(factory.list_stock_pools())

    # 尝试获取一个因子
    try:
        df = factory.get_factor('bid_ask_spread', 'high_frequency')
        print(f"\nbid_ask_spread 数据形状: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"获取因子数据失败: {e}")
