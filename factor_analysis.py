"""
高频因子分析框架
对已计算的高频因子进行多维度分析

功能模块:
1. 因子统计描述 (Descriptive Statistics)
2. 因子相关性分析 (Correlation Analysis)
3. 因子IC分析 (Information Coefficient)
4. 因子分层分析 (Factor Ranking Analysis)
5. 因子稳定性分析 (Stability Analysis)
6. 因子预测能力评估 (Predictive Power)
7. 可视化分析 (Visualization)
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')

# 可视化库
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns


class FactorAnalysis:
    """高频因子分析框架"""

    def __init__(self, factor_dir: str = "./factor/daily"):
        """
        初始化分析框架

        Args:
            factor_dir: 因子文件目录
        """
        self.factor_dir = Path(factor_dir)
        self.factors_df = None
        self.factor_cols = None
        self.stock_returns = None

    # ==================== 数据加载 ====================

    def load_all_factors(self, start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载所有因子数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票列表

        Returns:
            合并的因子数据
        """
        all_dfs = []

        for f in sorted(self.factor_dir.glob("factors_*.parquet")):
            file_date = self._parse_date_from_filename(f.name)

            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            df = pd.read_parquet(f)
            all_dfs.append(df)

        if not all_dfs:
            print("No factor data found!")
            return pd.DataFrame()

        self.factors_df = pd.concat(all_dfs, ignore_index=True)
        self.factors_df = self.factors_df.sort_values(['stock_code', 'datetime'])

        # 过滤股票
        if stock_codes:
            self.factors_df = self.factors_df[self.factors_df['stock_code'].isin(stock_codes)]

        # 定义因子列
        self.factor_cols = [c for c in self.factors_df.columns
                          if c not in ['time', 'datetime', 'lastPrice', 'volume',
                                      'amount', 'stock_code', 'date']]

        print(f"Loaded {len(self.factors_df):,} records, "
              f"{self.factors_df['stock_code'].nunique()} stocks, "
              f"{len(self.factor_cols)} factors")

        return self.factors_df

    def _parse_date_from_filename(self, filename: str) -> date:
        """从文件名解析日期"""
        date_str = filename.replace("factors_", "").replace(".parquet", "")
        return datetime.strptime(date_str, "%Y%m%d").date()

    def compute_returns(self, price_col: str = 'lastPrice',
                       periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        计算未来收益率（向量化优化版本）

        Args:
            price_col: 价格列名
            periods: 持有期列表

        Returns:
            包含未来收益率的数据
        """
        if self.factors_df is None:
            raise ValueError("Please load factors first!")

        df = self.factors_df.copy()
        df = df.sort_values(['stock_code', 'datetime'])

        # 使用向量化shift操作计算未来收益率
        for period in periods:
            # 计算未来收益率: (P(t+period) - P(t)) / P(t)
            future_price = df.groupby('stock_code')[price_col].shift(-period)
            current_price = df[price_col]
            df[f'return_{period}'] = (future_price - current_price) / current_price
            # 处理除以零的情况
            df[f'return_{period}'] = df[f'return_{period}'].where(current_price > 0, 0)

        self.factors_df = df
        return df

    # ==================== 描述性统计 ====================

    def descriptive_stats(self, groupby: Optional[str] = None) -> pd.DataFrame:
        """
        因子描述性统计

        Args:
            groupby: 分组字段（如 'stock_code', 'date'）

        Returns:
            统计信息DataFrame
        """
        if self.factors_df is None or self.factor_cols is None:
            raise ValueError("No data loaded!")

        stats = []

        if groupby:
            for name, group in self.factors_df.groupby(groupby):
                for col in self.factor_cols:
                    data = group[col].dropna()
                    if len(data) > 0:
                        stats.append({
                            'group': str(name),
                            'factor': col,
                            'count': len(data),
                            'mean': data.mean(),
                            'std': data.std(),
                            'min': data.min(),
                            'max': data.max(),
                            'skew': data.skew(),
                            'kurtosis': data.kurtosis(),
                            'ic': self._compute_ic(data, group[f'return_1'].dropna()) if 'return_1' in group.columns else np.nan
                        })
        else:
            for col in self.factor_cols:
                data = self.factors_df[col].dropna()
                if len(data) > 0:
                    stats.append({
                        'factor': col,
                        'count': len(data),
                        'mean': data.mean(),
                        'std': data.std(),
                        'min': data.min(),
                        'max': data.max(),
                        'skew': data.skew(),
                        'kurtosis': data.kurtosis()
                    })

        return pd.DataFrame(stats)

    def _compute_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """计算信息系数"""
        if len(factor_values) != len(returns):
            returns = returns.iloc[:len(factor_values)]
        return factor_values.corr(returns) if len(factor_values) > 10 else np.nan

    # ==================== 相关性分析 ====================

    def factor_correlation(self, method: str = 'spearman',
                          plot_path: Optional[str] = None) -> pd.DataFrame:
        """
        因子相关性分析

        Args:
            method: 相关性方法 ('pearson', 'spearman', 'kendall')
            plot_path: 图表保存路径

        Returns:
            相关性矩阵
        """
        if self.factors_df is None or self.factor_cols is None:
            raise ValueError("No data loaded!")

        # 计算相关性
        corr_matrix = self.factors_df[self.factor_cols].corr(method=method)

        # 绘制热力图
        if plot_path:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                        center=0, vmin=-1, vmax=1, square=True)
            plt.title(f'Factor Correlation Matrix ({method.capitalize()})')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved correlation heatmap to {plot_path}")

        return corr_matrix

    def time_decay_correlation(self, window: int = 20) -> pd.DataFrame:
        """
        时间衰减相关性分析

        Args:
            window: 滚动窗口大小

        Returns:
            滚动相关性DataFrame
        """
        if self.factors_df is None:
            raise ValueError("No data loaded!")

        dates = sorted(self.factors_df['date'].unique())
        results = []

        for i in range(window, len(dates)):
            window_dates = dates[i-window:i+1]
            window_df = self.factors_df[self.factors_df['date'].isin(window_dates)]

            for col in self.factor_cols[:3]:  # 只分析前3个因子
                for ret_col in [c for c in window_df.columns if c.startswith('return_')]:
                    ic = window_df[col].corr(window_df[ret_col])
                    results.append({
                        'date': dates[i],
                        'factor': col,
                        'return_horizon': ret_col,
                        'rolling_ic': ic
                    })

        return pd.DataFrame(results)

    # ==================== IC分析 ====================

    def ic_analysis(self, return_col: str = 'return_1',
                   by: str = 'date') -> Dict[str, pd.DataFrame]:
        """
        信息系数分析

        Args:
            return_col: 收益率列名
            by: 分组方式 ('date', 'stock_code')

        Returns:
            IC统计信息字典
        """
        if self.factors_df is None or self.factor_cols is None:
            raise ValueError("No data loaded!")

        if return_col not in self.factors_df.columns:
            raise ValueError(f"Column {return_col} not found! Compute returns first.")

        ic_stats = {}
        ic_series = {}

        for factor in self.factor_cols:
            if by == 'date':
                # 按日期计算IC
                date_ics = []
                for d, group in self.factors_df.groupby('date'):
                    valid = group[[factor, return_col]].dropna()
                    if len(valid) > 10:
                        ic = valid[factor].corr(valid[return_col])
                        date_ics.append({'date': d, 'ic': ic})
                ic_df = pd.DataFrame(date_ics)
            else:
                # 按股票计算IC
                stock_ics = []
                for s, group in self.factors_df.groupby('stock_code'):
                    valid = group[[factor, return_col]].dropna()
                    if len(valid) > 10:
                        ic = valid[factor].corr(valid[return_col])
                        stock_ics.append({'stock': s, 'ic': ic})
                ic_df = pd.DataFrame(stock_ics)

            ic_series[factor] = ic_df['ic'] if 'ic' in ic_df else pd.Series([np.nan])

            ic_stats[factor] = {
                'ic_mean': ic_series[factor].mean(),
                'ic_std': ic_series[factor].std(),
                'ic_ir': ic_series[factor].mean() / ic_series[factor].std() if ic_series[factor].std() > 0 else np.nan,
                'ic_positive_ratio': (ic_series[factor] > 0).mean(),
                'ic_t_stat': ic_series[factor].mean() / (ic_series[factor].std() / np.sqrt(len(ic_series[factor]))) if len(ic_series[factor]) > 1 else np.nan
            }

        return {
            'ic_series': ic_series,
            'ic_stats': pd.DataFrame(ic_stats).T
        }

    def ic_decay_analysis(self, max_lag: int = 20) -> pd.DataFrame:
        """
        IC衰减分析

        Args:
            max_lag: 最大滞后期数

        Returns:
            IC衰减数据
        """
        if self.factors_df is None:
            raise ValueError("No data loaded!")

        results = []

        for lag in range(1, max_lag + 1):
            ret_col = f'return_{lag}'
            if ret_col not in self.factors_df.columns:
                continue

            for factor in self.factor_cols:
                valid = self.factors_df[[factor, ret_col]].dropna()
                if len(valid) > 100:
                    ic = valid[factor].corr(valid[ret_col])
                    results.append({
                        'lag': lag,
                        'factor': factor,
                        'ic': ic
                    })

        return pd.DataFrame(results)

    # ==================== 分层分析 ====================

    def quantile_analysis(self, factor: str, return_col: str = 'return_1',
                         quantiles: int = 5, plot_path: Optional[str] = None) -> pd.DataFrame:
        """
        分层回测分析

        Args:
            factor: 因子名称
            return_col: 收益率列名
            quantiles: 分组数量
            plot_path: 图表保存路径

        Returns:
            分组收益统计
        """
        if self.factors_df is None:
            raise ValueError("No data loaded!")

        df = self.factors_df[[factor, return_col, 'date']].dropna()

        if len(df) == 0:
            return pd.DataFrame()

        # 按因子值分组
        df['quantile'] = pd.qcut(df[factor], q=quantiles, labels=False, duplicates='drop')

        # 计算每组收益
        results = []
        for q in range(quantiles):
            group = df[df['quantile'] == q]
            results.append({
                'quantile': q + 1,
                'factor_mean': group[factor].mean(),
                'return_mean': group[return_col].mean(),
                'return_std': group[return_col].std(),
                'count': len(group),
                'ic': group[factor].corr(group[return_col]) if len(group) > 10 else np.nan
            })

        result_df = pd.DataFrame(results)

        # 绘图
        if plot_path:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 平均收益
            axes[0].bar(result_df['quantile'], result_df['return_mean'] * 100,
                       color=['red' if x < 0 else 'green' for x in result_df['return_mean']])
            axes[0].set_xlabel('Quantile')
            axes[0].set_ylabel('Mean Return (%)')
            axes[0].set_title(f'{factor} - Mean Return by Quantile')
            axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # IC
            axes[1].bar(result_df['quantile'], result_df['ic'])
            axes[1].set_xlabel('Quantile')
            axes[1].set_ylabel('IC')
            axes[1].set_title(f'{factor} - IC by Quantile')

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved quantile analysis to {plot_path}")

        return result_df

    def long_short_portfolio(self, factor: str, return_col: str = 'return_1',
                            plot_path: Optional[str] = None) -> pd.DataFrame:
        """
        多空组合分析

        Args:
            factor: 因子名称
            return_col: 收益率列名
            plot_path: 图表保存路径

        Returns:
            多空收益统计
        """
        df = self.factors_df[[factor, return_col, 'date']].dropna()
        df['quantile'] = pd.qcut(df[factor], q=5, labels=False, duplicates='drop')

        # 计算每日多空收益
        daily_ls = []
        for d, group in df.groupby('date'):
            top = group[group['quantile'] == 4][return_col].mean()
            bottom = group[group['quantile'] == 0][return_col].mean()
            ls_ret = top - bottom
            daily_ls.append({'date': d, 'ls_return': ls_ret})

        ls_df = pd.DataFrame(daily_ls)

        stats = {
            'total_return': ls_df['ls_return'].sum(),
            'mean_daily': ls_df['ls_return'].mean(),
            'std_daily': ls_df['ls_return'].std(),
            'sharpe': ls_df['ls_return'].mean() / ls_df['ls_return'].std() * np.sqrt(252) if ls_df['ls_return'].std() > 0 else 0,
            'win_rate': (ls_df['ls_return'] > 0).mean(),
            'max_drawdown': self._compute_drawdown(ls_df['ls_return'].cumsum())
        }

        if plot_path:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            cumulative = ls_df['ls_return'].cumsum()
            plt.plot(pd.to_datetime(ls_df['date']), cumulative)
            plt.title(f'{factor} - Long Short Cumulative Return')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            plt.subplot(1, 2, 2)
            plt.hist(ls_df['ls_return'], bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'{factor} - Daily Return Distribution')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved long-short analysis to {plot_path}")

        return pd.DataFrame([stats])

    def _compute_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = returns.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        return drawdown.min()

    # ==================== 稳定性分析 ====================

    def factor_stability(self, window: int = 10) -> pd.DataFrame:
        """
        因子稳定性分析

        Args:
            window: 滚动窗口大小

        Returns:
            稳定性统计
        """
        if self.factors_df is None:
            raise ValueError("No data loaded!")

        dates = sorted(self.factors_df['date'].unique())
        results = []

        for factor in self.factor_cols:
            factor_stability = []
            for i in range(window, len(dates)):
                window_dates = dates[i-window:i+1]
                window_df = self.factors_df[self.factors_df['date'].isin(window_dates)]

                # 计算组内IC的std（稳定性）
                date_ics = []
                for d, g in window_df.groupby('date'):
                    valid = g[[factor, 'return_1']].dropna()
                    if len(valid) > 5:
                        ic = valid[factor].corr(valid['return_1'])
                        date_ics.append(ic)

                if len(date_ics) > 1:
                    factor_stability.append({
                        'date': dates[i],
                        'factor': factor,
                        'ic_std': np.std(date_ics),
                        'ic_mean': np.mean(date_ics)
                    })

            if factor_stability:
                results.extend(factor_stability)

        return pd.DataFrame(results)

    def turnover_analysis(self, groupby: str = 'stock_code') -> pd.DataFrame:
        """
        因子换手率分析

        Args:
            groupby: 分组方式

        Returns:
            换手率统计
        """
        if self.factors_df is None:
            raise ValueError("No data loaded!")

        turnover = {}
        dates = sorted(self.factors_df['date'].unique())

        for factor in self.factor_cols[:3]:  # 分析前3个因子
            factor_turnover = []
            for i in range(1, len(dates)):
                prev_df = self.factors_df[self.factors_df['date'] == dates[i-1]]
                curr_df = self.factors_df[self.factors_df['date'] == dates[i]]

                if groupby == 'stock':
                    # 计算股票维度换手
                    prev_ranks = prev_df.groupby('stock_code')[factor].mean().rank()
                    curr_ranks = curr_df.groupby('stock_code')[factor].mean().rank()

                    # 计算排名变化
                    common = set(prev_ranks.index) & set(curr_ranks.index)
                    if len(common) > 0:
                        rank_change = (curr_ranks[common] - prev_ranks[common]).abs().mean()
                        factor_turnover.append({'date': dates[i], 'turnover': rank_change})
                else:
                    # 计算整体换手
                    factor_turnover.append({
                        'date': dates[i],
                        'turnover': 1 - (prev_df[factor].corr(curr_df[factor]))
                    })

            turnover[factor] = pd.DataFrame(factor_turnover)

        return turnover

    # ==================== 综合分析报告 ====================

    def full_analysis_report(self, output_dir: str = "./factor_analysis",
                            return_col: str = 'return_1') -> Dict[str, Any]:
        """
        生成完整分析报告

        Args:
            output_dir: 输出目录
            return_col: 收益率列名

        Returns:
            分析结果字典
        """
        os.makedirs(output_dir, exist_ok=True)

        report = {}

        print("=" * 60)
        print("Generating Factor Analysis Report")
        print("=" * 60)

        # 1. 描述性统计
        print("\n1. Descriptive Statistics...")
        desc_stats = self.descriptive_stats()
        desc_stats.to_csv(f"{output_dir}/descriptive_stats.csv", index=False)
        report['descriptive_stats'] = desc_stats

        # 2. 相关性分析
        print("\n2. Factor Correlation Analysis...")
        corr = self.factor_correlation(method='spearman',
                                      plot_path=f"{output_dir}/correlation_heatmap.png")
        corr.to_csv(f"{output_dir}/factor_correlation.csv")
        report['correlation'] = corr

        # 3. IC分析
        print("\n3. Information Coefficient Analysis...")
        ic_result = self.ic_analysis(return_col=return_col, by='date')
        ic_result['ic_stats'].to_csv(f"{output_dir}/ic_statistics.csv")
        report['ic_stats'] = ic_result['ic_stats']

        # 4. IC衰减分析
        print("\n4. IC Decay Analysis...")
        ic_decay = self.ic_decay_analysis(max_lag=20)
        ic_decay.to_csv(f"{output_dir}/ic_decay.csv", index=False)
        report['ic_decay'] = ic_decay

        # 绘制IC衰减图
        if not ic_decay.empty:
            plt.figure(figsize=(10, 6))
            for factor in ic_decay['factor'].unique():
                factor_decay = ic_decay[ic_decay['factor'] == factor]
                plt.plot(factor_decay['lag'], factor_decay['ic'], marker='o', label=factor)
            plt.xlabel('Lag (Days)')
            plt.ylabel('IC')
            plt.title('IC Decay Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/ic_decay_plot.png", dpi=150, bbox_inches='tight')
            plt.close()

        # 5. 分层分析（对前3个因子）
        print("\n5. Quantile Analysis...")
        for factor in self.factor_cols[:3]:
            quant_result = self.quantile_analysis(factor, return_col,
                                                  plot_path=f"{output_dir}/quantile_{factor}.png")
            quant_result.to_csv(f"{output_dir}/quantile_{factor}.csv", index=False)

        # 6. 多空组合分析
        print("\n6. Long-Short Portfolio Analysis...")
        ls_results = []
        for factor in self.factor_cols[:3]:
            ls = self.long_short_portfolio(factor, return_col,
                                           plot_path=f"{output_dir}/ls_{factor}.png")
            ls['factor'] = factor
            ls_results.append(ls)
        ls_df = pd.concat(ls_results, ignore_index=True)
        ls_df.to_csv(f"{output_dir}/long_short_analysis.csv", index=False)
        report['long_short'] = ls_df

        # 7. 保存汇总
        summary = {
            'total_records': len(self.factors_df),
            'total_stocks': self.factors_df['stock_code'].nunique(),
            'total_dates': self.factors_df['date'].nunique(),
            'factor_count': len(self.factor_cols),
            'top_factor_by_ic': ic_result['ic_stats']['ic_mean'].idxmax() if 'ic_mean' in ic_result['ic_stats'].columns else None,
            'best_ic': ic_result['ic_stats']['ic_mean'].max() if 'ic_mean' in ic_result['ic_stats'].columns else None,
            'best_ir': ic_result['ic_stats']['ic_ir'].max() if 'ic_ir' in ic_result['ic_stats'].columns else None
        }
        pd.DataFrame([summary]).to_csv(f"{output_dir}/analysis_summary.csv", index=False)
        report['summary'] = summary

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)

        return report


# ==================== 使用示例 ====================

def run_factor_analysis():
    """运行因子分析示例"""
    # 初始化分析框架
    analyzer = FactorAnalysis("./factor/daily")

    # 加载因子数据
    analyzer.load_all_factors()

    # 计算未来收益率
    analyzer.compute_returns(periods=[1, 5, 10])

    # 运行完整分析
    report = analyzer.full_analysis_report(
        output_dir="./factor_analysis_results",
        return_col='return_1'
    )

    # 打印汇总
    print("\n" + "=" * 60)
    print("Factor Analysis Summary")
    print("=" * 60)
    for key, value in report['summary'].items():
        print(f"{key}: {value}")

    return report


if __name__ == "__main__":
    run_factor_analysis()
