"""
中证1000高频因子分析
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns


class HighFrequencyFactorAnalyzer:
    """高频因子分析器"""

    def __init__(self, factor_dir: str = "./factor/daily", output_dir: str = "./hf_analysis_results"):
        self.factor_dir = Path(factor_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_factors = None
        self.daily_factors = None
        self.returns = None
        self.ic_results = None
        self.ranking_results = None
        self.ic_all_periods = None

        # 高频因子列表
        self.hf_factor_cols = [
            'order_imbalance', 'effective_spread', 'realized_volatility',
            'bid_ask_spread', 'vwap_deviation', 'price_momentum',
            'trade_flow_intensity', 'micro_price', 'trade_imbalance',
            'depth_imbalance'
        ]

    def load_all_factors(self):
        """加载所有高频因子数据"""
        print("\n" + "="*60)
        print("加载高频因子数据")
        print("="*60)

        factor_files = sorted(self.factor_dir.glob("zz1000_factors_*.parquet"))
        print(f"发现 {len(factor_files)} 个因子文件")

        all_dfs = []
        for i, f in enumerate(factor_files):
            if (i + 1) % 5 == 0:
                print(f"  加载: {i+1}/{len(factor_files)}")
            df = pd.read_parquet(f)
            all_dfs.append(df)

        self.raw_factors = pd.concat(all_dfs, ignore_index=True)

        # 过滤存在的因子
        self.hf_factor_cols = [c for c in self.hf_factor_cols if c in self.raw_factors.columns]

        print(f"\n原始数据: {len(self.raw_factors):,} 条记录")
        print(f"股票数量: {self.raw_factors['stock_code'].nunique():,}")
        print(f"日期范围: {self.raw_factors['date'].min()} ~ {self.raw_factors['date'].max()}")
        print(f"\n高频因子列表 ({len(self.hf_factor_cols)}个):")
        for col in self.hf_factor_cols:
            print(f"  - {col}")

        return self

    def aggregate_daily_factors(self):
        """将高频因子聚合为日度因子"""
        print("\n" + "="*60)
        print("聚合日度因子")
        print("="*60)

        df = self.raw_factors.copy()

        # 按股票和日期聚合
        agg_dict = {}
        for col in self.hf_factor_cols:
            agg_dict[col] = ['mean', 'std']

        daily = df.groupby(['stock_code', 'date']).agg(agg_dict)
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        daily = daily.reset_index()

        # 添加价格信息
        price_info = df.groupby(['stock_code', 'date']).agg({
            'lastPrice': ['first', 'last'],
            'open': 'first'
        }).reset_index()
        price_info.columns = ['stock_code', 'date', 'first_price', 'last_price', 'open']

        self.daily_factors = pd.merge(daily, price_info, on=['stock_code', 'date'], how='left')

        # 计算日收益率
        self.daily_factors['daily_return'] = (
            self.daily_factors['last_price'] / self.daily_factors['first_price']
        ) - 1

        print(f"日度因子: {len(self.daily_factors):,} 条记录")
        print(f"股票数量: {self.daily_factors['stock_code'].nunique():,}")

        return self

    def calculate_future_returns(self, holding_periods: list = [1, 5, 10, 20]):
        """计算未来收益率"""
        print("\n" + "="*60)
        print("计算未来收益率")
        print("="*60)

        df = self.daily_factors.copy()
        df = df.sort_values(['stock_code', 'date'])

        # 获取因子列
        factor_cols = [c for c in df.columns if any(hf in c for hf in self.hf_factor_cols)]

        for period in holding_periods:
            # 计算未来period天的累计收益率
            df[f'future_return_{period}d'] = df.groupby('stock_code')['daily_return'].transform(
                lambda x: x.shift(-1).rolling(period).sum().shift(-period + 1)
            )

        self.returns = df[['stock_code', 'date', 'daily_return'] +
                          [f'future_return_{period}d' for period in holding_periods]].copy()

        # 合并因子
        self.merged_data = pd.merge(
            df[['stock_code', 'date'] + factor_cols],
            self.returns,
            on=['stock_code', 'date'],
            how='inner'
        )

        print(f"合并数据: {len(self.merged_data):,} 条记录")

        return self

    def calculate_ic(self, return_col: str = 'future_return_1d'):
        """计算IC"""
        print("\n" + "="*60)
        print(f"因子IC分析 (持有期: {return_col})")
        print("="*60)

        df = self.merged_data.copy()

        # 获取因子列
        factor_cols = [c for c in df.columns if any(hf in c for hf in self.hf_factor_cols)]

        ic_results = {}

        for col in factor_cols:
            # 简化列名
            factor_name = col.split('_mean')[0] if '_mean' in col else col
            if factor_name not in self.hf_factor_cols:
                continue

            valid_data = df[['stock_code', col, return_col]].dropna()

            if len(valid_data) < 100:
                continue

            # 计算IC
            ic = valid_data[col].corr(valid_data[return_col])
            n = len(valid_data)
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            spearman_ic, _ = stats.spearmanr(valid_data[col], valid_data[return_col])

            ic_results[factor_name] = {
                'IC': ic,
                'RankIC': spearman_ic,
                'IC_abs': abs(ic),
                'IC_pvalue': p_value,
                'n_samples': len(valid_data)
            }

        self.ic_results = pd.DataFrame(ic_results).T.sort_values('IC_abs', ascending=False)

        print("\nIC分析结果:")
        print("-" * 75)
        print(f"{'因子':<22} {'IC':>10} {'RankIC':>10} {'|IC|':>10} {'P值':>12} {'显著性':>8}")
        print("-" * 75)

        for idx, row in self.ic_results.iterrows():
            sig = "***" if row['IC_pvalue'] < 0.001 else "**" if row['IC_pvalue'] < 0.01 else "*" if row['IC_pvalue'] < 0.05 else ""
            print(f"{idx:<22} {row['IC']:>10.4f} {row['RankIC']:>10.4f} {row['IC_abs']:>10.4f} {row['IC_pvalue']:>12.4f} {sig:>8}")

        return self.ic_results

    def calculate_ic_all_periods(self):
        """计算不同持有期的IC"""
        print("\n" + "="*60)
        print("多持有期IC分析")
        print("="*60)

        periods = ['future_return_1d', 'future_return_5d', 'future_return_10d', 'future_return_20d']
        all_ic = {}

        for period in periods:
            df = self.merged_data.copy()

            for factor in self.hf_factor_cols:
                col = f"{factor}_mean"
                if col not in df.columns:
                    continue

                key = f"{factor}_{period.replace('future_return_', '')}"

                valid_data = df[['stock_code', col, period]].dropna()
                if len(valid_data) < 100:
                    continue

                ic = valid_data[col].corr(valid_data[period])
                all_ic[key] = ic

        self.ic_all_periods = pd.DataFrame(all_ic, index=[0]).T
        self.ic_all_periods.columns = ['IC']

        # 打印透视表
        pivot_data = []
        for k, v in all_ic.items():
            factor, period = k.rsplit('_', 1)
            pivot_data.append({'factor': factor, 'period': period.replace('d', ''), 'IC': v})

        pivot_df = pd.DataFrame(pivot_data)
        pivot = pivot_df.pivot(index='factor', columns='period', values='IC')

        print("\n各因子在不同持有期的IC:")
        print(pivot.round(4))

        return self.ic_all_periods

    def run_all_ranking_analysis(self, return_col: str = 'future_return_1d', n_groups: int = 5):
        """分层回测"""
        print("\n" + "="*60)
        print("因子分层回测分析")
        print("="*60)

        df = self.merged_data.copy()

        ranking_results = []

        for factor in self.hf_factor_cols:
            col = f"{factor}_mean"
            if col not in df.columns:
                continue

            valid_data = df[['stock_code', col, return_col]].dropna()

            if len(valid_data) < 100:
                continue

            try:
                valid_data['group'] = pd.qcut(valid_data[col], q=n_groups, labels=False, duplicates='drop')
                group_returns = valid_data.groupby('group')[return_col].agg(['mean', 'std', 'count']).reset_index()

                ls_return = group_returns.iloc[-1]['mean'] - group_returns.iloc[0]['mean']
                direction = '正向' if group_returns.iloc[-1]['mean'] > group_returns.iloc[0]['mean'] else '负向'

                ic_val = self.ic_results.loc[factor, 'IC'] if factor in self.ic_results.index else 0

                ranking_results.append({
                    '因子': factor,
                    'IC': ic_val,
                    '多空收益': ls_return * 100,
                    '方向': direction,
                    'Q1收益': group_returns.iloc[0]['mean'] * 100,
                    'Q5收益': group_returns.iloc[-1]['mean'] * 100
                })
            except Exception as e:
                continue

        self.ranking_results = pd.DataFrame(ranking_results).sort_values('多空收益', ascending=False)

        print("\n分层回测结果:")
        print("-" * 85)
        print(f"{'因子':<22} {'IC':>8} {'多空收益':>12} {'方向':>6} {'Q1收益':>10} {'Q5收益':>10}")
        print("-" * 85)

        for _, row in self.ranking_results.iterrows():
            print(f"{row['因子']:<22} {row['IC']:>8.4f} {row['多空收益']:>10.2f}% {row['方向']:>6} {row['Q1收益']:>9.2f}% {row['Q5收益']:>9.2f}%")

        return self.ranking_results

    def visualize_results(self):
        """可视化"""
        if self.ic_results is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. IC柱状图
        ax1 = axes[0, 0]
        ic_sorted = self.ic_results.sort_values('IC')
        colors = ['green' if x > 0 else 'red' for x in ic_sorted['IC']]
        ax1.barh(range(len(ic_sorted)), ic_sorted['IC'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(ic_sorted)))
        ax1.set_yticklabels(ic_sorted.index, fontsize=9)
        ax1.set_xlabel('IC')
        ax1.set_title('Factor Information Coefficient (IC)')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 2. |IC|排名
        ax2 = axes[0, 1]
        ic_abs = self.ic_results.sort_values('IC_abs', ascending=True)
        ax2.barh(range(len(ic_abs)), ic_abs['IC_abs'], color='steelblue', alpha=0.7)
        ax2.set_yticks(range(len(ic_abs)))
        ax2.set_yticklabels(ic_abs.index, fontsize=9)
        ax2.set_xlabel('|IC|')
        ax2.set_title('Factor |IC| Ranking')
        ax2.axvline(x=0.02, color='red', linestyle='--', alpha=0.7, label='Weak (0.02)')
        ax2.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7, label='Medium (0.05)')
        ax2.legend(fontsize=8)

        # 3. 多持有期IC热力图
        ax3 = axes[1, 0]
        if hasattr(self, 'ic_all_periods') and self.ic_all_periods is not None:
            pivot_data = []
            for k, v in self.ic_all_periods['IC'].items():
                factor, period = k.rsplit('_', 1)
                pivot_data.append({'factor': factor, 'period': period, 'IC': v})
            pivot_df = pd.DataFrame(pivot_data)
            pivot = pivot_df.pivot(index='factor', columns='period', values='IC')
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=0, ax=ax3)
            ax3.set_title('IC Across Different Holding Periods')

        # 4. 分层回测
        ax4 = axes[1, 1]
        if hasattr(self, 'ranking_results') and self.ranking_results is not None:
            top10 = self.ranking_results.head(10)
            colors = ['green' if x > 0 else 'red' for x in top10['多空收益']]
            ax4.barh(range(len(top10)), top10['多空收益'], color=colors, alpha=0.7)
            ax4.set_yticks(range(len(top10)))
            ax4.set_yticklabels(top10['因子'], fontsize=9)
            ax4.set_xlabel('Long-Short Return (%)')
            ax4.set_title('Top 10 Factors by Long-Short Return')
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'hf_factor_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n分析图已保存: {self.output_dir / 'hf_factor_analysis.png'}")

    def save_results(self):
        """保存结果"""
        print("\n" + "="*60)
        print("保存分析结果")
        print("="*60)

        if self.ic_results is not None:
            self.ic_results.to_csv(self.output_dir / 'hf_ic_results.csv')
            print(f"IC结果: {self.output_dir / 'hf_ic_results.csv'}")

        if hasattr(self, 'ranking_results') and self.ranking_results is not None:
            self.ranking_results.to_csv(self.output_dir / 'hf_ranking_results.csv', index=False)
            print(f"分层结果: {self.output_dir / 'hf_ranking_results.csv'}")

        if hasattr(self, 'ic_all_periods') and self.ic_all_periods is not None:
            self.ic_all_periods.to_csv(self.output_dir / 'hf_ic_all_periods.csv')
            print(f"多持有期IC: {self.output_dir / 'hf_ic_all_periods.csv'}")

    def run_full_analysis(self):
        """完整分析流程"""
        print("\n" + "="*80)
        print("中证1000高频因子分析 - 开始")
        print("="*80)

        self.load_all_factors()
        self.aggregate_daily_factors()
        self.calculate_future_returns()
        self.calculate_ic('future_return_1d')
        self.calculate_ic_all_periods()
        self.run_all_ranking_analysis()
        self.visualize_results()
        self.save_results()

        print("\n" + "="*80)
        print("分析完成!")
        print("="*80)

        return self


# ==================== 主程序 ====================

if __name__ == "__main__":
    analyzer = HighFrequencyFactorAnalyzer(
        factor_dir="/data1/code_git/tick_data_analysis/factor/daily",
        output_dir="./hf_analysis_results"
    )
    analyzer.run_full_analysis()

    # 打印总结
    print("\n" + "="*80)
    print("核心发现")
    print("="*80)

    print("\n【最强正向因子】(因子值高 → 未来收益高)")
    top_pos = analyzer.ic_results[analyzer.ic_results['IC'] > 0].head(3)
    for idx, row in top_pos.iterrows():
        print(f"  {idx}: IC={row['IC']:.4f}")

    print("\n【负向因子】(因子值高 → 未来收益低)")
    top_neg = analyzer.ic_results[analyzer.ic_results['IC'] < 0].head(3)
    for idx, row in top_neg.iterrows():
        print(f"  {idx}: IC={row['IC']:.4f}")

    print("\n" + "="*80)
    print("策略建议")
    print("="*80)
    print("""
1. 【反转效应】高频领域价格倾向于短期反转
   - high realized_volatility → low return
   - high bid_ask_spread → low return (流动性溢价)

2. 【订单流信号】
   - order_imbalance: 反映短期供需
   - trade_imbalance: 主动买卖力量对比

3. 【日内交易建议】
   - 观察vwap_deviation回归
   - 利用price_momentum反转
   - 筛选低spread、低volatility标的
""")
