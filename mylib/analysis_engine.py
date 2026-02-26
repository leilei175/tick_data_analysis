"""
分析引擎模块
封装完整的因子分析流程
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import pandas as pd
import numpy as np

# 添加父目录到路径
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from mylib.factor_factory import FactorFactory
from mylib.factor_preprocessor import FactorPreprocessor
from mylib.returns_calculator import ReturnsCalculator
from mylib.get_local_data import get_local_data


class AnalysisConfig:
    """分析配置类"""

    def __init__(
        self,
        factor_name: str,
        source: str = 'high_frequency',
        stock_pool: str = 'zz1000',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        preprocess_method: Optional[str] = None,
        preprocess_params: Optional[Dict] = None,
        returns_method: str = 'close2close_next',
        returns_n: int = 5,
        quantiles: int = 5,
        filters: Optional[Dict] = None
    ):
        self.factor_name = factor_name
        self.source = source
        self.stock_pool = stock_pool
        self.start_date = start_date
        self.end_date = end_date
        self.preprocess_method = preprocess_method
        self.preprocess_params = preprocess_params or {}
        self.returns_method = returns_method
        self.returns_n = returns_n
        self.quantiles = quantiles
        self.filters = filters or {}

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'source': self.source,
            'stock_pool': self.stock_pool,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'preprocess_method': self.preprocess_method,
            'preprocess_params': self.preprocess_params,
            'returns_method': self.returns_method,
            'returns_n': self.returns_n,
            'quantiles': self.quantiles,
            'filters': self.filters
        }


class AnalysisResult:
    """分析结果类"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.ic_stats: Optional[Dict] = None
        self.quantile_returns: Optional[List[Dict]] = None
        self.long_short_stats: Optional[Dict] = None
        self.turnover_rate: Optional[float] = None
        self.correlation_matrix: Optional[Dict] = None
        self.ic_series: Optional[List[Dict]] = None
        self.charts_data: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'config': self.config.to_dict(),
            'created_at': self.created_at,
            'ic_stats': self.ic_stats,
            'quantile_returns': self.quantile_returns,
            'long_short_stats': self.long_short_stats,
            'turnover_rate': self.turnover_rate,
            'correlation_matrix': self.correlation_matrix,
            'ic_series': self.ic_series,
            'charts_data': self.charts_data
        }


class AnalysisEngine:
    """
    分析引擎

    封装完整的因子分析流程：
    1. 加载因子数据
    2. 预处理因子
    3. 计算收益率
    4. 计算IC、分层收益、多空组合等指标
    """

    def __init__(
        self,
        factor_dir: Optional[str] = None,
        preprocessed_dir: Optional[str] = None,
        returns_dir: Optional[str] = None,
        analysis_dir: Optional[str] = None
    ):
        """
        初始化分析引擎

        Args:
            factor_dir: 因子数据目录
            preprocessed_dir: 预处理数据目录
            returns_dir: 收益率数据目录
            analysis_dir: 分析结果保存目录
        """
        self.factory = FactorFactory(factor_dir)
        self.preprocessor = FactorPreprocessor(preprocessed_dir)
        self.calculator = ReturnsCalculator(returns_dir)

        self.analysis_dir = Path(analysis_dir) if analysis_dir else Path(_parent_dir) / 'factor' / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self, config: AnalysisConfig) -> AnalysisResult:
        """
        执行因子分析

        Args:
            config: 分析配置

        Returns:
            AnalysisResult: 分析结果
        """
        print(f"开始分析: {config.factor_name} ({config.source})")

        result = AnalysisResult(config)

        try:
            # 1. 加载因子数据
            print("  [1/5] 加载因子数据...")
            factor_df = self.factory.get_factor(
                factor_name=config.factor_name,
                source=config.source,
                stock_pool=config.stock_pool,
                start_date=config.start_date,
                end_date=config.end_date
            )
            if factor_df.empty:
                raise ValueError("因子数据为空，请调整起止日期或检查因子文件")
            print(f"       因子数据形状: {factor_df.shape}")

            # 2. 预处理因子
            if config.preprocess_method:
                print(f"  [2/5] 预处理因子 ({config.preprocess_method})...")
                factor_df = self.preprocessor.preprocess(
                    factor_df,
                    config.preprocess_method,
                    **config.preprocess_params
                )
                # 保存预处理后的因子
                self.preprocessor.save(
                    factor_df,
                    config.factor_name,
                    config.preprocess_method,
                    config.stock_pool,
                    **config.preprocess_params
                )
                print(f"       预处理后数据形状: {factor_df.shape}")

            # 3. 加载价格并计算收益率
            print(f"  [3/5] 计算收益率 ({config.returns_method})...")
            start_date = config.start_date or factor_df.index.min().strftime('%Y%m%d')
            end_date = config.end_date or factor_df.index.max().strftime('%Y%m%d')

            target_stocks = list(factor_df.columns)
            close_df = get_local_data(
                sec_list=target_stocks,
                start=start_date,
                end=end_date,
                filed='close',
                data_type='daily'
            )
            open_df = get_local_data(
                sec_list=target_stocks,
                start=start_date,
                end=end_date,
                filed='open',
                data_type='daily'
            )
            if close_df.empty:
                raise ValueError("无法加载日线收盘价数据，无法计算收益率")

            returns_df = self.calculator.calculate(
                close_df=close_df,
                open_df=open_df if not open_df.empty else None,
                method=config.returns_method,
                n=config.returns_n
            )
            print(f"       收益率数据形状: {returns_df.shape}")

            # 4. 对齐数据
            print("  [4/5] 对齐数据...")
            common_dates = factor_df.index.intersection(returns_df.index)
            common_stocks = factor_df.columns.intersection(returns_df.columns)
            if len(common_dates) == 0 or len(common_stocks) == 0:
                raise ValueError("因子与收益率对齐后无可用样本，请检查日期范围和股票池")

            factor_aligned = factor_df.loc[common_dates, common_stocks]
            returns_aligned = returns_df.loc[common_dates, common_stocks]

            print(f"       对齐后形状: {factor_aligned.shape}")
            print(f"       日期范围: {common_dates.min()} ~ {common_dates.max()}")

            # 5. 计算分析指标
            print("  [5/5] 计算分析指标...")

            # 计算IC
            ic_series = self._calc_ic_series(factor_aligned, returns_aligned)
            result.ic_series = [{'date': str(d), 'ic': float(v)} for d, v in ic_series.items()]
            result.ic_stats = self._calc_ic_stats(ic_series)

            # 计算分层收益
            result.quantile_returns = self._calc_quantile_returns(
                factor_aligned, returns_aligned, config.quantiles
            )
            quantile_daily = self._calc_quantile_daily_returns(
                factor_aligned, returns_aligned, config.quantiles
            )

            # 计算多空组合
            result.long_short_stats = self._calc_long_short(
                factor_aligned, returns_aligned, config.quantiles
            )

            # 计算换手率
            result.turnover_rate = self._calc_turnover_rate(factor_aligned)

            # 准备图表数据
            result.charts_data = {
                'ic_series': result.ic_series,
                'quantile_returns': result.quantile_returns,
                'quantile_daily': quantile_daily,
                'long_short': result.long_short_stats
            }

            print("       分析完成!")

        except Exception as e:
            print(f"       错误: {e}")
            raise

        return result

    def _calc_ic_series(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame
    ) -> pd.Series:
        """
        计算IC序列

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据

        Returns:
            pd.Series: IC序列
        """
        ic_series = []

        for date in factor_df.index:
            factor_day = factor_df.loc[date].dropna()
            returns_day = returns_df.loc[date].dropna()

            # 取交集
            common = factor_day.index.intersection(returns_day.index)

            if len(common) > 10:
                factor_vals = factor_day.loc[common]
                returns_vals = returns_day.loc[common]

                # 计算截面相关系数
                ic = factor_vals.corr(returns_vals)
                if not np.isnan(ic):
                    ic_series.append((date, ic))

        return pd.Series(dict(ic_series))

    def _calc_ic_stats(self, ic_series: pd.Series) -> Dict:
        """
        计算IC统计信息

        Args:
            ic_series: IC序列

        Returns:
            Dict: IC统计信息
        """
        ic_values = ic_series.dropna()

        if len(ic_values) == 0:
            return {
                'ic_mean': 0,
                'ic_std': 0,
                'ic_ir': 0,
                'ic_positive_ratio': 0,
                'ic_count': 0
            }

        ic_mean = float(ic_values.mean())
        ic_std = float(ic_values.std())
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_ratio': float((ic_values > 0).mean()),
            'ic_count': len(ic_values)
        }

    def _calc_quantile_returns(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        quantiles: int = 5
    ) -> List[Dict]:
        """
        计算分层收益

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            quantiles: 分层数量

        Returns:
            List[Dict]: 分层收益信息
        """
        results = []

        for q in range(1, quantiles + 1):
            q_returns = []

            for date in factor_df.index:
                factor_day = factor_df.loc[date].dropna()
                returns_day = returns_df.loc[date].dropna()

                common = factor_day.index.intersection(returns_day.index)

                if len(common) > 10:
                    factor_vals = factor_day.loc[common]
                    returns_vals = returns_day.loc[common]

                    # 计算分位数阈值
                    try:
                        thresholds = pd.qcut(factor_vals, quantiles, labels=False, duplicates='drop')
                        mask = thresholds == (q - 1)

                        if mask.sum() > 0:
                            q_returns.extend(returns_vals.loc[mask].tolist())
                    except:
                        continue

            if q_returns:
                q_returns_arr = np.array(q_returns)
                results.append({
                    'quantile': q,
                    'mean': float(np.mean(q_returns_arr)),
                    'std': float(np.std(q_returns_arr)),
                    'count': len(q_returns_arr),
                    'sharpe': float(np.mean(q_returns_arr) / np.std(q_returns_arr) * np.sqrt(252)) if np.std(q_returns_arr) > 0 else 0
                })
            else:
                results.append({
                    'quantile': q,
                    'mean': 0,
                    'std': 0,
                    'count': 0,
                    'sharpe': 0
                })

        return results

    def _calc_long_short(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        quantiles: int = 5
    ) -> Dict:
        """
        计算多空组合收益

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            quantiles: 分层数量

        Returns:
            Dict: 多空组合统计
        """
        top_returns = []
        bottom_returns = []
        ls_returns = []

        for date in factor_df.index:
            factor_day = factor_df.loc[date].dropna()
            returns_day = returns_df.loc[date].dropna()

            common = factor_day.index.intersection(returns_day.index)

            if len(common) > 10:
                factor_vals = factor_day.loc[common]
                returns_vals = returns_day.loc[common]

                try:
                    thresholds = pd.qcut(factor_vals, quantiles, labels=False, duplicates='drop')

                    top_mask = thresholds == (quantiles - 1)  # 最高因子值
                    bottom_mask = thresholds == 0  # 最低因子值

                    if top_mask.sum() > 0 and bottom_mask.sum() > 0:
                        top_mean = float(returns_vals.loc[top_mask].mean())
                        bottom_mean = float(returns_vals.loc[bottom_mask].mean())
                        top_returns.append(top_mean)
                        bottom_returns.append(bottom_mean)
                        ls_returns.append(top_mean - bottom_mean)
                except:
                    continue

        if not ls_returns:
            return {
                'total_return': 0,
                'mean_daily': 0,
                'std_daily': 0,
                'sharpe': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'count': 0
            }

        ls_arr = np.array(ls_returns, dtype=float)
        if len(ls_arr) > 0:
            cumulative = np.cumprod(1 + ls_arr)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max

            return {
                'total_return': float((1 + ls_arr).prod() - 1),
                'mean_daily': float(np.mean(ls_arr)),
                'std_daily': float(np.std(ls_arr)),
                'sharpe': float(np.mean(ls_arr) / np.std(ls_arr) * np.sqrt(252)) if np.std(ls_arr) > 0 else 0,
                'win_rate': float((ls_arr > 0).mean()),
                'max_drawdown': float(drawdown.min()),
                'count': len(ls_arr)
            }

        return {
            'total_return': 0,
            'mean_daily': 0,
            'std_daily': 0,
            'sharpe': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'count': 0
        }

    def _calc_quantile_daily_returns(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        quantiles: int = 5
    ) -> List[Dict]:
        """
        计算每个分层的日度收益序列

        Returns:
            List[Dict]: [{'date': 'YYYY-MM-DD', 'Q1': x, ...}, ...]
        """
        rows: List[Dict] = []

        for date in factor_df.index:
            factor_day = factor_df.loc[date].dropna()
            returns_day = returns_df.loc[date].dropna()
            common = factor_day.index.intersection(returns_day.index)

            if len(common) <= 10:
                continue

            factor_vals = factor_day.loc[common]
            returns_vals = returns_day.loc[common]

            try:
                buckets = pd.qcut(factor_vals, quantiles, labels=False, duplicates='drop')
            except Exception:
                continue

            row = {'date': pd.Timestamp(date).strftime('%Y-%m-%d')}
            valid_bucket_count = int(buckets.max()) + 1 if len(buckets) > 0 else 0
            for q in range(quantiles):
                key = f"Q{q + 1}"
                if q < valid_bucket_count:
                    mask = buckets == q
                    if int(mask.sum()) > 0:
                        row[key] = float(returns_vals.loc[mask].mean())
                    else:
                        row[key] = 0.0
                else:
                    row[key] = 0.0

            rows.append(row)

        return rows

    def _calc_turnover_rate(self, factor_df: pd.DataFrame) -> float:
        """
        估算因子换手率

        Args:
            factor_df: 因子数据

        Returns:
            float: 平均换手率
        """
        # 通过因子排名变化估算换手率
        turnover_rates = []

        for i in range(1, len(factor_df)):
            prev_rank = factor_df.iloc[i - 1].rank()
            curr_rank = factor_df.iloc[i].rank()

            # 计算排名变化
            rank_diff = (prev_rank - curr_rank).abs()

            # 换手率 = 排名发生显著变化的比例
            turnover = (rank_diff > len(factor_df.columns) * 0.1).mean()
            turnover_rates.append(turnover)

        return float(np.mean(turnover_rates)) if turnover_rates else 0

    def save_result(
        self,
        result: AnalysisResult,
        output_dir: Optional[str] = None
    ) -> str:
        """
        保存分析结果

        Args:
            result: 分析结果
            output_dir: 输出目录

        Returns:
            str: 保存路径
        """
        output_dir = Path(output_dir) if output_dir else self.analysis_dir

        # 生成目录名
        date_str = datetime.now().strftime('%Y%m%d')
        method_str = result.config.preprocess_method or 'raw'
        dir_name = f"{date_str}_{result.config.factor_name}_{method_str}"

        result_dir = output_dir / dir_name
        result_dir.mkdir(parents=True, exist_ok=True)

        # 保存IC统计
        if result.ic_stats:
            ic_df = pd.DataFrame([result.ic_stats])
            ic_df.to_csv(result_dir / 'ic_stats.csv', index=False)

        # 保存分层收益
        if result.quantile_returns:
            quantile_df = pd.DataFrame(result.quantile_returns)
            quantile_df.to_csv(result_dir / 'quantile_returns.csv', index=False)

        # 保存完整报告
        report_path = result_dir / 'full_report.md'
        self._generate_report(result, report_path)

        return str(result_dir)

    def _generate_report(self, result: AnalysisResult, report_path: Path):
        """
        生成分析报告

        Args:
            result: 分析结果
            report_path: 报告路径
        """
        config = result.config

        report = f"""# 因子分析报告

## 基本信息

- **因子名称**: {config.factor_name}
- **因子来源**: {config.source}
- **股票池**: {config.stock_pool}
- **分析日期**: {result.created_at}
- **日期范围**: {config.start_date} ~ {config.end_date}

## 预处理配置

- **预处理方法**: {config.preprocess_method or '无'}
- **预处理参数**: {config.preprocess_params}

## 收益率配置

- **收益率计算方法**: {config.returns_method}
- **持有天数**: {config.returns_n}

## IC分析结果

| 指标 | 值 |
|------|------|
| IC均值 | {result.ic_stats['ic_mean']:.4f} |
| IC标准差 | {result.ic_stats['ic_std']:.4f} |
| IC信息比率 | {result.ic_stats['ic_ir']:.4f} |
| IC正向比例 | {result.ic_stats['ic_positive_ratio']:.2%} |
| 样本数量 | {result.ic_stats['ic_count']} |

## 分层收益分析

| 分层 | 均值 | 标准差 | 夏普比率 | 样本数 |
|------|------|--------|----------|--------|
"""

        for qr in result.quantile_returns:
            report += f"| Q{qr['quantile']} | {qr['mean']:.4f} | {qr['std']:.4f} | {qr['sharpe']:.4f} | {qr['count']} |\n"

        report += f"""
## 多空组合分析

| 指标 | 值 |
|------|------|
| 总收益 | {result.long_short_stats['total_return']:.2%} |
| 日均收益 | {result.long_short_stats['mean_daily']:.4f} |
| 日波动率 | {result.long_short_stats['std_daily']:.4f} |
| 夏普比率 | {result.long_short_stats['sharpe']:.4f} |
| 胜率 | {result.long_short_stats['win_rate']:.2%} |
| 最大回撤 | {result.long_short_stats['max_drawdown']:.2%} |

## 换手率

- **平均换手率**: {result.turnover_rate:.2%}

---
*报告生成时间: {result.created_at}*
"""

        report_path.write_text(report, encoding='utf-8')


# 全局分析引擎实例
_default_engine: Optional[AnalysisEngine] = None


def get_analysis_engine(
    factor_dir: Optional[str] = None,
    preprocessed_dir: Optional[str] = None,
    returns_dir: Optional[str] = None,
    analysis_dir: Optional[str] = None
) -> AnalysisEngine:
    """
    获取全局分析引擎实例

    Args:
        factor_dir: 因子数据目录
        preprocessed_dir: 预处理数据目录
        returns_dir: 收益率数据目录
        analysis_dir: 分析结果目录

    Returns:
        AnalysisEngine: 分析引擎实例
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = AnalysisEngine(
            factor_dir=factor_dir,
            preprocessed_dir=preprocessed_dir,
            returns_dir=returns_dir,
            analysis_dir=analysis_dir
        )
    return _default_engine


if __name__ == '__main__':
    # 测试代码
    engine = get_analysis_engine()

    # 创建测试配置
    config = AnalysisConfig(
        factor_name='bid_ask_spread',
        source='high_frequency',
        stock_pool='zz1000',
        start_date='20260101',
        end_date='20260131',
        preprocess_method='zscore',
        preprocess_params={'std_threshold': 3},
        returns_method='close2close_next',
        quantiles=5
    )

    print("分析配置:")
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    print("\n可用因子来源:")
    print(engine.factory.list_sources())

    print("\n可用预处理方法:")
    print(list(engine.preprocessor.list_methods().keys()))

    print("\n可用收益率计算方法:")
    print(list(engine.calculator.list_methods().keys()))

    # 尝试运行分析
    try:
        result = engine.run_analysis(config)
        print("\n分析结果:")
        print(f"  IC均值: {result.ic_stats['ic_mean']:.4f}")
        print(f"  IC信息比率: {result.ic_stats['ic_ir']:.4f}")
        print(f"  多空夏普: {result.long_short_stats['sharpe']:.4f}")

        # 保存结果
        save_path = engine.save_result(result)
        print(f"\n结果已保存至: {save_path}")
    except FileNotFoundError as e:
        print(f"\n测试数据不存在，跳过实际分析: {e}")
    except Exception as e:
        print(f"\n分析失败: {e}")
