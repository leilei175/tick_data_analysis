"""
收益率计算器模块
提供多种收益率计算方法
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

# 添加父目录到路径
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


class ReturnsCalculator:
    """
    收益率计算器

    支持以下收益率计算方法（均与T日因子对齐）：
    - open2close: 次日开盘价到次日收盘价收益率
    - open2open_next: 次日开盘价到T+2日开盘价收益率
    - close2close_next: 次日收盘价到T+2日收盘价收益率
    - close2close_n: 次日收盘价到T+N日收盘价收益率
    """

    # 收益率计算方法配置
    METHODS = {
        'open2close': {
            'name': '次日开-次日收',
            'description': 'T日因子对应次日开盘价到次日收盘价收益率',
            'params': {
                'period': {'type': 'int', 'default': 1, 'desc': '保留参数（兼容）'}
            }
        },
        'open2open_next': {
            'name': '次日开-T+2开',
            'description': 'T日因子对应次日开盘价到T+2日开盘价收益率',
            'params': {
                'shift': {'type': 'int', 'default': 1, 'desc': '向前天数（兼容）'}
            }
        },
        'close2close_next': {
            'name': '次日收-T+2收',
            'description': 'T日因子对应次日收盘价到T+2日收盘价收益率',
            'params': {
                'shift': {'type': 'int', 'default': 1, 'desc': '向前天数（兼容）'}
            }
        },
        'close2close_n': {
            'name': '次日收-T+N收',
            'description': 'T日因子对应次日收盘价到T+N日收盘价收益率（N建议3/5/10）',
            'params': {
                'n': {'type': 'int', 'default': 5, 'desc': '持有终点（交易日）'}
            }
        }
    }

    def __init__(self, returns_dir: Optional[str] = None):
        """
        初始化收益率计算器

        Args:
            returns_dir: 收益率数据保存目录
        """
        self.base_dir = Path(returns_dir) if returns_dir else Path(_parent_dir) / 'factor' / 'returns'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}

    def get_base_dir(self) -> Path:
        """获取收益率数据保存目录"""
        return self.base_dir

    def list_methods(self) -> Dict:
        """列出所有可用的收益率计算方法"""
        return self.METHODS

    def calculate(
        self,
        close_df: pd.DataFrame,
        open_df: Optional[pd.DataFrame] = None,
        method: str = 'close2close_next',
        n: int = 5
    ) -> pd.DataFrame:
        """
        计算收益率

        Args:
            close_df: 收盘价数据（宽格式）
            open_df: 开盘价数据（宽格式），可选
            method: 计算方法
            n: 持有天数（仅对close2close_n有效）

        Returns:
            pd.DataFrame: 收益率数据（宽格式）
        """
        if method == 'open2close':
            return self._calc_open2close(close_df, open_df, n)
        if method == 'open2open_next':
            return self._calc_open2open_next(close_df, open_df)
        if method == 'close2close_next':
            return self._calc_close2close_next(close_df)
        if method == 'close2close_n':
            return self._calc_close2close_n(close_df, n)

        raise ValueError(f"未知收益率计算方法: {method}，可用方法: {list(self.METHODS.keys())}")

    def _calc_open2close(
        self,
        close_df: pd.DataFrame,
        open_df: Optional[pd.DataFrame] = None,
        period: int = 1
    ) -> pd.DataFrame:
        """
        计算次日开盘到次日收盘收益率

        R_t = C_{t+1} / O_{t+1} - 1

        Args:
            close_df: 收盘价数据
            open_df: 开盘价数据
            period: 保留参数（兼容）

        Returns:
            pd.DataFrame: 收益率数据
        """
        if open_df is None:
            open_df = close_df
        return close_df.shift(-1) / open_df.shift(-1) - 1

    def _calc_open2open_next(
        self,
        close_df: pd.DataFrame,
        open_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        计算次日开盘到T+2日开盘收益率

        R_t = O_{t+2} / O_{t+1} - 1

        Args:
            close_df: 收盘价数据
            open_df: 开盘价数据

        Returns:
            pd.DataFrame: 收益率数据
        """
        if open_df is None:
            open_df = close_df
        return open_df.shift(-2) / open_df.shift(-1) - 1

    def _calc_close2close_next(
        self,
        close_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算次日收盘到T+2日收盘收益率

        R_t = C_{t+2} / C_{t+1} - 1

        Args:
            close_df: 收盘价数据

        Returns:
            pd.DataFrame: 收益率数据
        """
        return close_df.shift(-2) / close_df.shift(-1) - 1

    def _calc_close2close_n(
        self,
        close_df: pd.DataFrame,
        n: int = 5
    ) -> pd.DataFrame:
        """
        计算次日收盘到T+N日收盘收益率

        R_t = C_{t+N} / C_{t+1} - 1

        Args:
            close_df: 收盘价数据
            n: 终点日（建议3/5/10）

        Returns:
            pd.DataFrame: 收益率数据
        """
        n = max(int(n), 2)
        return close_df.shift(-n) / close_df.shift(-1) - 1

    def calculate_from_price(
        self,
        prices: pd.DataFrame,
        method: str = 'close2close_next',
        **kwargs
    ) -> pd.DataFrame:
        """
        根据价格数据计算收益率

        Args:
            prices: 价格数据，必须包含'close'列，可选'open'列
            method: 计算方法
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 收益率数据
        """
        close_df = prices['close']

        if 'open' in prices.columns:
            open_df = prices['open']
        else:
            open_df = None

        return self.calculate(close_df, open_df, method, **kwargs)

    def get_returns_file(
        self,
        method: str,
        n: int = 5,
        stock_pool: str = 'zz1000'
    ) -> Optional[Path]:
        """
        获取已保存的收益率文件路径

        Args:
            method: 计算方法
            n: 持有天数
            stock_pool: 股票池

        Returns:
            Path: 文件路径，不存在返回None
        """
        if method == 'close2close_n':
            filename = f"return_{method}_{n}d_{stock_pool}.parquet"
        else:
            filename = f"return_{method}_{stock_pool}.parquet"

        filepath = self.base_dir / filename

        if filepath.exists():
            return filepath

        return None

    def save_returns(
        self,
        returns_df: pd.DataFrame,
        method: str,
        n: int = 5,
        stock_pool: str = 'zz1000'
    ) -> str:
        """
        保存收益率数据

        Args:
            returns_df: 收益率数据
            method: 计算方法
            n: 持有天数
            stock_pool: 股票池

        Returns:
            str: 保存的文件路径
        """
        if method == 'close2close_n':
            filename = f"return_{method}_{n}d_{stock_pool}.parquet"
        else:
            filename = f"return_{method}_{stock_pool}.parquet"

        filepath = self.base_dir / filename
        returns_df.to_parquet(filepath)

        return str(filepath)

    def load_returns(
        self,
        method: str,
        n: int = 5,
        stock_pool: str = 'zz1000'
    ) -> Optional[pd.DataFrame]:
        """
        加载收益率数据

        Args:
            method: 计算方法
            n: 持有天数
            stock_pool: 股票池

        Returns:
            pd.DataFrame: 收益率数据，不存在返回None
        """
        filepath = self.get_returns_file(method, n, stock_pool)

        if filepath:
            return pd.read_parquet(filepath)

        return None

    def list_saved_returns(self) -> List[Dict]:
        """
        列出已保存的收益率文件

        Returns:
            List: 文件信息列表
        """
        files = list(self.base_dir.glob('*.parquet'))

        result = []
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            result.append({
                'filename': f.name,
                'file_size_mb': round(size_mb, 2),
                'file_path': str(f)
            })

        return result


# 全局收益率计算器实例
_default_calculator: Optional[ReturnsCalculator] = None


def get_calculator(returns_dir: Optional[str] = None) -> ReturnsCalculator:
    """
    获取全局收益率计算器实例

    Args:
        returns_dir: 收益率数据保存目录

    Returns:
        ReturnsCalculator: 收益率计算器实例
    """
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = ReturnsCalculator(returns_dir)
    return _default_calculator


if __name__ == '__main__':
    # 测试代码
    calculator = ReturnsCalculator()

    print("可用收益率计算方法:")
    for method, config in calculator.list_methods().items():
        print(f"  - {method}: {config['name']}")
        print(f"    {config['description']}")

    # 创建测试价格数据
    dates = pd.date_range('20260101', periods=100, freq='D')
    stocks = [f'{i:06d}.SZ' for i in range(50)]

    np.random.seed(42)
    close_data = np.random.randn(100, 50) * 0.02 + 0.01
    close_df = pd.DataFrame(
        close_data.cumsum(axis=0) + 100,
        index=dates,
        columns=stocks
    )

    print("\n测试收益率计算 (close2close_next):")
    returns = calculator.calculate(close_df, method='close2close_next')
    print(f"  收益率形状: {returns.shape}")
    print(f"  均值: {returns.values[~np.isnan(returns.values)].mean():.6f}")
    print(f"  标准差: {returns.values[~np.isnan(returns.values)].std():.6f}")

    print("\n测试收益率计算 (close2close_n, n=5):")
    returns_5d = calculator.calculate(close_df, method='close2close_n', n=5)
    print(f"  5日收益率形状: {returns_5d.shape}")
