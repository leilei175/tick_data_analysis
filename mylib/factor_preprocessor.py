"""
因子预处理器模块
提供多种因子预处理方法
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats

# 添加父目录到路径
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


class FactorPreprocessor:
    """
    因子预处理器

    支持以下预处理方法：
    - zscore: 标准化（均值0，标准差1）
    - winsorize: 去极值（指定分位数截断）
    - normalize: 归一化（0-1范围）
    - rank: 排序映射（转为排名百分位）
    - neutralize: 中性化（对行业/市值回归取残差）
    """

    # 预处理方法配置
    METHODS = {
        'zscore': {
            'name': '标准化',
            'description': '将因子转换为均值为0、标准差为1的分布',
            'params': {
                'std_threshold': {'type': 'float', 'default': 3.0, 'range': [1.0, 5.0], 'desc': 'Z-score截断阈值'}
            }
        },
        'winsorize': {
            'name': '去极值',
            'description': '将超出指定分位数的值截断到边界',
            'params': {
                'lower_quantile': {'type': 'float', 'default': 0.025, 'range': [0.0, 0.1], 'desc': '下侧分位数'},
                'upper_quantile': {'type': 'float', 'default': 0.975, 'range': [0.9, 1.0], 'desc': '上侧分位数'}
            }
        },
        'normalize': {
            'name': '归一化',
            'description': '将因子归一化到0-1范围',
            'params': {}
        },
        'rank': {
            'name': '排序映射',
            'description': '将因子值转换为排名百分位',
            'params': {}
        },
        'neutralize': {
            'name': '中性化',
            'description': '对行业和市值因子回归，取残差进行中性化',
            'params': {
                'market_cap': {'type': 'bool', 'default': True, 'desc': '是否对市值中性化'},
                'industry': {'type': 'bool', 'default': True, 'desc': '是否对行业中性化'}
            }
        },
        'mad': {
            'name': 'MAD去极值',
            'description': '使用MAD（绝对中位差）方法去极值',
            'params': {
                'threshold': {'type': 'float', 'default': 5.0, 'range': [3.0, 10.0], 'desc': 'MAD倍数阈值'}
            }
        },
        'log': {
            'name': '对数变换',
            'description': '对因子值进行对数变换',
            'params': {
                'add_constant': {'type': 'bool', 'default': True, 'desc': '是否添加常数避免log(0)'}
            }
        }
    }

    def __init__(self, preprocessed_dir: Optional[str] = None):
        """
        初始化预处理器

        Args:
            preprocessed_dir: 预处理后数据保存目录
        """
        self.base_dir = Path(preprocessed_dir) if preprocessed_dir else Path(_parent_dir) / 'factor' / 'preprocessed'
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_base_dir(self) -> Path:
        """获取预处理数据保存目录"""
        return self.base_dir

    def list_methods(self) -> Dict:
        """列出所有可用的预处理方法"""
        return self.METHODS

    def preprocess(
        self,
        factor_df: pd.DataFrame,
        method: str,
        **params
    ) -> pd.DataFrame:
        """
        对因子数据进行预处理

        Args:
            factor_df: 宽格式因子数据
            method: 预处理方法
            **params: 方法参数

        Returns:
            pd.DataFrame: 预处理后的因子数据
        """
        if method == 'zscore':
            return self._zscore(factor_df, **params)
        elif method == 'winsorize':
            return self._winsorize(factor_df, **params)
        elif method == 'normalize':
            return self._normalize(factor_df, **params)
        elif method == 'rank':
            return self._rank(factor_df)
        elif method == 'neutralize':
            return self._neutralize(factor_df, **params)
        elif method == 'mad':
            return self._mad(factor_df, **params)
        elif method == 'log':
            return self._log_transform(factor_df, **params)
        else:
            raise ValueError(f"未知预处理方法: {method}，可用方法: {list(self.METHODS.keys())}")

    def _zscore(
        self,
        factor_df: pd.DataFrame,
        std_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Z-score标准化

        Args:
            factor_df: 因子数据
            std_threshold: Z-score截断阈值

        Returns:
            pd.DataFrame: 标准化后的数据
        """
        result = factor_df.copy()

        # 计算截面Z-score
        for date in result.index:
            date_data = result.loc[date].dropna()
            if len(date_data) > 1:
                mean = date_data.mean()
                std = date_data.std()

                if std > 0:
                    z_scores = (date_data - mean) / std
                    # 截断极端值
                    z_scores = z_scores.clip(-std_threshold, std_threshold)
                    # 还原到原始尺度
                    result.loc[date, z_scores.index] = z_scores * std + mean

        # 最终标准化到均值0标准差1
        flat_values = result.values.flatten()
        flat_values = flat_values[~np.isnan(flat_values)]
        mean = np.mean(flat_values)
        std = np.std(flat_values)

        if std > 0:
            result = (result - mean) / std

        return result

    def _winsorize(
        self,
        factor_df: pd.DataFrame,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975
    ) -> pd.DataFrame:
        """
        去极值（分位数截断）

        Args:
            factor_df: 因子数据
            lower_quantile: 下侧分位数
            upper_quantile: 上侧分位数

        Returns:
            pd.DataFrame: 去极值后的数据
        """
        result = factor_df.copy()

        for date in result.index:
            date_data = result.loc[date].dropna()
            if len(date_data) > 1:
                lower_bound = date_data.quantile(lower_quantile)
                upper_bound = date_data.quantile(upper_quantile)

                clipped = date_data.clip(lower=lower_bound, upper=upper_bound)
                result.loc[date, clipped.index] = clipped

        return result

    def _normalize(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        归一化（0-1范围）

        Args:
            factor_df: 因子数据

        Returns:
            pd.DataFrame: 归一化后的数据
        """
        result = factor_df.copy()

        for date in result.index:
            date_data = result.loc[date].dropna()
            if len(date_data) > 1:
                min_val = date_data.min()
                max_val = date_data.max()

                if max_val > min_val:
                    normalized = (date_data - min_val) / (max_val - min_val)
                    result.loc[date, normalized.index] = normalized

        return result

    def _rank(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        排序映射（转换为排名百分位）

        Args:
            factor_df: 因子数据

        Returns:
            pd.DataFrame: 排名百分位数据
        """
        result = factor_df.copy()

        for date in result.index:
            date_data = result.loc[date].dropna()
            if len(date_data) > 1:
                # 计算排名百分位
                ranks = date_data.rank(pct=True)
                result.loc[date, ranks.index] = ranks

        return result

    def _neutralize(
        self,
        factor_df: pd.DataFrame,
        market_cap: bool = True,
        industry: bool = True
    ) -> pd.DataFrame:
        """
        中性化（对行业/市值回归取残差）

        注意：这是一个简化版本，实际应用中需要提供行业和市值数据

        Args:
            factor_df: 因子数据
            market_cap: 是否对市值中性化
            industry: 是否对行业中性化

        Returns:
            pd.DataFrame: 中性化后的数据
        """
        result = factor_df.copy()

        # 如果没有额外的行业/市值数据，只进行截面标准化
        print("注意: 中性化需要额外的行业和市值数据，当前使用简化的截面标准化")

        for date in result.index:
            date_data = result.loc[date].dropna()
            if len(date_data) > 1:
                # 简化的中性化：减去行业均值（这里用分组均值模拟）
                mean = date_data.mean()
                result.loc[date, date_data.index] = date_data - mean

        return result

    def _mad(
        self,
        factor_df: pd.DataFrame,
        threshold: float = 5.0
    ) -> pd.DataFrame:
        """
        MAD去极值（绝对中位差）

        Args:
            factor_df: 因子数据
            threshold: MAD倍数阈值

        Returns:
            pd.DataFrame: 去极值后的数据
        """
        result = factor_df.copy()

        for date in result.index:
            date_data = result.loc[date].dropna()
            if len(date_data) > 1:
                median = date_data.median()
                mad = np.median(np.abs(date_data - median))

                if mad > 0:
                    modified_z = 0.6745 * (date_data - median) / mad
                    clipped = date_data.clip(
                        lower=median - threshold * mad,
                        upper=median + threshold * mad
                    )
                    result.loc[date, clipped.index] = clipped

        return result

    def _log_transform(
        self,
        factor_df: pd.DataFrame,
        add_constant: bool = True
    ) -> pd.DataFrame:
        """
        对数变换

        Args:
            factor_df: 因子数据
            add_constant: 是否添加常数避免log(0)

        Returns:
            pd.DataFrame: 对数变换后的数据
        """
        result = factor_df.copy()

        flat_values = result.values.flatten()
        flat_values = flat_values[~np.isnan(flat_values)]

        min_val = np.min(flat_values)
        if add_constant and min_val <= 0:
            offset = abs(min_val) + 1
        else:
            offset = 0

        result = np.log(result + offset)

        return result

    def preview(
        self,
        factor_df: pd.DataFrame,
        method: str,
        **params
    ) -> Dict:
        """
        预览预处理效果

        Args:
            factor_df: 因子数据（样本）
            method: 预处理方法
            **params: 方法参数

        Returns:
            Dict: 预处理前后的统计对比
        """
        # 原始数据统计
        flat_before = factor_df.values.flatten()
        flat_before = flat_before[~np.isnan(flat_before)]

        before_stats = {
            'count': int(len(flat_before)),
            'mean': float(np.mean(flat_before)),
            'std': float(np.std(flat_before)),
            'min': float(np.min(flat_before)),
            'max': float(np.max(flat_before)),
            'median': float(np.median(flat_before)),
            'skew': float(pd.Series(flat_before).skew()),
            'kurtosis': float(pd.Series(flat_before).kurtosis())
        }

        # 应用预处理
        after_df = self.preprocess(factor_df, method, **params)

        # 预处理后统计
        flat_after = after_df.values.flatten()
        flat_after = flat_after[~np.isnan(flat_after)]

        after_stats = {
            'count': int(len(flat_after)),
            'mean': float(np.mean(flat_after)),
            'std': float(np.std(flat_after)),
            'min': float(np.min(flat_after)),
            'max': float(np.max(flat_after)),
            'median': float(np.median(flat_after)),
            'skew': float(pd.Series(flat_after).skew()),
            'kurtosis': float(pd.Series(flat_after).kurtosis())
        }

        return {
            'method': method,
            'params': params,
            'before_stats': before_stats,
            'after_stats': after_stats
        }

    def save(
        self,
        factor_df: pd.DataFrame,
        factor_name: str,
        method: str,
        stock_pool: str = 'zz1000',
        **params
    ) -> str:
        """
        保存预处理后的因子数据

        Args:
            factor_df: 预处理后的因子数据
            factor_name: 原始因子名称
            method: 预处理方法
            stock_pool: 股票池
            **params: 方法参数

        Returns:
            str: 保存的文件路径
        """
        # 生成文件名
        param_str = '_'.join(f"{k[0:3]}{v}" for k, v in sorted(params.items()))
        if param_str:
            filename = f"pp_{stock_pool}_{factor_name}_{method}_{param_str}.parquet"
        else:
            filename = f"pp_{stock_pool}_{factor_name}_{method}.parquet"

        filepath = self.base_dir / filename
        factor_df.to_parquet(filepath)

        return str(filepath)

    def load(
        self,
        factor_name: str,
        method: str,
        stock_pool: str = 'zz1000',
        **params
    ) -> Optional[pd.DataFrame]:
        """
        加载预处理后的因子数据

        Args:
            factor_name: 因子名称
            method: 预处理方法
            stock_pool: 股票池
            **params: 方法参数

        Returns:
            pd.DataFrame: 预处理后的因子数据，不存在返回None
        """
        param_str = '_'.join(f"{k[0:3]}{v}" for k, v in sorted(params.items()))
        if param_str:
            filename = f"pp_{stock_pool}_{factor_name}_{method}_{param_str}.parquet"
        else:
            filename = f"pp_{stock_pool}_{factor_name}_{method}.parquet"

        filepath = self.base_dir / filename

        if filepath.exists():
            return pd.read_parquet(filepath)

        return None


# 全局预处理器实例
_default_preprocessor: Optional[FactorPreprocessor] = None


def get_preprocessor(preprocessed_dir: Optional[str] = None) -> FactorPreprocessor:
    """
    获取全局预处理器实例

    Args:
        preprocessed_dir: 预处理数据保存目录

    Returns:
        FactorPreprocessor: 预处理器实例
    """
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = FactorPreprocessor(preprocessed_dir)
    return _default_preprocessor


if __name__ == '__main__':
    # 测试代码
    preprocessor = FactorPreprocessor()

    print("可用预处理方法:")
    for method, config in preprocessor.list_methods().items():
        print(f"  - {method}: {config['name']} - {config['description']}")

    # 创建测试数据
    test_data = pd.DataFrame(
        np.random.randn(100, 50),
        index=pd.date_range('20260101', periods=100, freq='D'),
        columns=[f'{i:06d}.SZ' for i in range(50)]
    )
    # 添加一些极端值
    test_data.iloc[0, 0] = 100
    test_data.iloc[1, 1] = -50

    print("\n预览预处理效果 (zscore):")
    result = preprocessor.preview(test_data, 'zscore', std_threshold=3)
    print(f"  原始均值: {result['before_stats']['mean']:.4f}, 预处理后均值: {result['after_stats']['mean']:.4f}")
    print(f"  原始标准差: {result['before_stats']['std']:.4f}, 预处理后标准差: {result['after_stats']['std']:.4f}")
