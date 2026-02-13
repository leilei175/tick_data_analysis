"""
中证1000基本面因子分析 - 改进版
根据财报披露日期将季度因子转换为日度因子

核心逻辑：
- 季度财报有公告日期(ann_date)
- 在公告日之前，因子值沿用上一季度的数据
- 在公告日之后，因子值更新为当前季度的数据
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
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

# 导入数据下载模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_financial_statements import get_income, get_balance, get_cashflow
from download_daily_basic import get_close
from download_daily_data import get_daily


class FundamentalFactorAnalyzerV2:
    """基本面因子分析器V2 - 使用日度因子数据"""

    def __init__(self, output_dir: str = "./fundamental_analysis_results_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.income = None
        self.balance = None
        self.cashflow = None
        self.daily = None
        self.daily_factors = None
        self.factors = None
        self.returns = None
        self.ic_results = None
        self.ranking_results = None

    # ==================== 数据加载 ====================

    def load_financial_data(self, start_period: str = '20240331', end_period: str = '20241231'):
        """加载财务报表数据"""
        print("\n" + "="*60)
        print("加载财务报表数据")
        print("="*60)

        print("加载利润表...")
        self.income = get_income(start_period=start_period, end_period=end_period)
        print(f"  利润表: {len(self.income):,} 条记录")

        print("加载资产负债表...")
        self.balance = get_balance(start_period=start_period, end_period=end_period)
        print(f"  资产负债表: {len(self.balance):,} 条记录")

        print("加载现金流量表...")
        self.cashflow = get_cashflow(start_period=start_period, end_period=end_period)
        print(f"  现金流量表: {len(self.cashflow):,} 条记录")

        return self

    def load_daily_data(self, start_day: str = '20250101', end_day: str = '20260206'):
        """加载日线数据"""
        print("\n加载日线数据...")
        self.daily = get_daily(start_day=start_day, end_day=end_day)
        print(f"  日线数据: {len(self.daily):,} 条记录, {self.daily['ts_code'].nunique():,} 只股票")
        return self

    # ==================== 数据合并 ====================

    def merge_financial_data(self):
        """合并三张财务报表"""
        print("\n" + "="*60)
        print("合并财务报表")
        print("="*60)

        # 合并利润表和资产负债表
        merged = pd.merge(
            self.income,
            self.balance,
            on=['ts_code', 'end_date', 'report_type', 'comp_type'],
            how='outer',
            suffixes=('_inc', '_bal')
        )

        # 合并现金流量表
        merged = pd.merge(
            merged,
            self.cashflow,
            on=['ts_code', 'end_date', 'report_type', 'comp_type'],
            how='outer'
        )

        # 去重，保留最新公告日期的数据
        merged = merged.sort_values('ann_date')
        merged = merged.drop_duplicates(subset=['ts_code', 'end_date'], keep='last')

        print(f"  合并后: {len(merged):,} 条记录, {merged['ts_code'].nunique():,} 只股票")

        # 转换日期格式
        merged['ann_date'] = pd.to_datetime(merged['ann_date'].astype(str))
        merged['end_date'] = pd.to_datetime(merged['end_date'].astype(str))

        self.merged_data = merged
        return self

    # ==================== 因子构建 ====================

    def build_quarterly_factors(self):
        """构建季度因子"""
        print("\n" + "="*60)
        print("构建季度因子")
        print("="*60)

        df = self.merged_data.copy()

        # 获取最新收盘价
        if self.daily is not None:
            latest_price = self.daily.groupby('ts_code')['close'].last().reset_index()
            latest_price.columns = ['ts_code', 'close_price']
            df = pd.merge(df, latest_price, on='ts_code', how='left')

        # ========== 1. 估值因子 ==========
        df['pe'] = df['close_price'] / (df['basic_eps'] + 1e-10)
        df['pb'] = df['close_price'] / (df['total_hldr_eqy_exc_min_int'] / 1e8 + 1e-10)
        df['ps'] = (df['close_price'] * 1e8) / (df['revenue'] + 1e-10)

        # ========== 2. 盈利能力因子 ==========
        df['net_profit_margin'] = df['n_income'] / (df['revenue'] + 1e-10)
        df['operating_margin'] = df['operate_profit'] / (df['revenue'] + 1e-10)
        df['roe'] = df['n_income'] / (df['total_hldr_eqy_exc_min_int'] + 1e-10)
        df['roa'] = df['n_income'] / (df['total_assets'] + 1e-10)
        df['basic_eps_factor'] = df['basic_eps']

        # ========== 3. 成长因子 ==========
        df = df.sort_values(['ts_code', 'end_date'])
        df['revenue_growth'] = df.groupby('ts_code')['revenue'].pct_change(periods=4)
        df['profit_growth'] = df.groupby('ts_code')['n_income'].pct_change(periods=4)
        df['eps_growth'] = df.groupby('ts_code')['basic_eps'].pct_change(periods=4)

        # ========== 4. 运营效率因子 ==========
        df['asset_turnover'] = df['revenue'] / (df['total_assets'] + 1e-10)
        df['ar_turnover'] = df['revenue'] / (df['accounts_receiv'] + 1e-10)
        df['inventory_turnover'] = df['revenue'] / (df['inventories'] + 1e-10)

        # ========== 5. 杠杆/偿债因子 ==========
        df['debt_ratio'] = df['total_liab'] / (df['total_assets'] + 1e-10)
        df['current_ratio'] = df['total_cur_assets'] / (df['total_cur_liab'] + 1e-10)
        df['quick_ratio'] = (df['total_cur_assets'] - df['inventories']) / (df['total_cur_liab'] + 1e-10)
        df['lt_debt_ratio'] = (df['total_liab'] - df['total_cur_liab']) / (df['total_liab'] + 1e-10)

        # ========== 6. 现金流因子 ==========
        df['ocf_to_netincome'] = df['n_cashflow_act'] / (df['n_income'] + 1e-10)
        df['ocf_yield'] = df['n_cashflow_act'] / (df['total_assets'] + 1e-10)
        df['inv_cash_ratio'] = df['n_cashflow_inv_act'] / (df['n_cashflow_act'] + 1e-10)

        # 因子列表
        self.factor_cols = [
            'pe', 'pb', 'ps',
            'net_profit_margin', 'operating_margin', 'roe', 'roa', 'basic_eps_factor',
            'revenue_growth', 'profit_growth', 'eps_growth',
            'asset_turnover', 'ar_turnover', 'inventory_turnover',
            'debt_ratio', 'current_ratio', 'quick_ratio', 'lt_debt_ratio',
            'ocf_to_netincome', 'ocf_yield', 'inv_cash_ratio'
        ]
        self.factor_cols = [c for c in self.factor_cols if c in df.columns]

        # 清理异常值
        for col in self.factor_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            q1, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(q1, q99)

        # 只保留需要的列
        self.quarterly_factors = df[['ts_code', 'ann_date', 'end_date'] + self.factor_cols].copy()

        print(f"  季度因子构建完成: {len(self.quarterly_factors):,} 条记录")
        return self

    # ==================== 核心：季度因子转日度因子 ====================

    def convert_to_daily_factors(self):
        """
        将季度因子根据披露日期转换为日度因子（高效版）

        核心逻辑：
        - 每只股票在每个交易日都有一个因子值
        - 在财报公告日(ann_date)之前，因子值沿用上一季度的数据
        - 在财报公告日之后，因子值更新为当前季度的数据

        高效方法：
        1. 将季度因子展开为每日（每个财报日对应一天）
        2. 使用merge_asof进行最近邻匹配
        3. 使用ffill向前填充
        """
        print("\n" + "="*60)
        print("将季度因子转换为日度因子（高效版）")
        print("="*60)

        if self.daily is None or self.quarterly_factors is None:
            print("错误: 需要先加载日线数据和季度因子")
            return self

        # 转换日期格式
        daily = self.daily.copy()
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str))
        daily = daily.sort_values(['ts_code', 'trade_date'])

        quarterly = self.quarterly_factors.copy()
        quarterly['ann_date'] = pd.to_datetime(quarterly['ann_date'])
        quarterly = quarterly.sort_values(['ts_code', 'ann_date'])

        print(f"  日线数据: {len(daily):,} 条")
        print(f"  季度因子: {len(quarterly):,} 条")

        # 获取所有股票
        stocks = daily['ts_code'].unique()
        print(f"  股票数量: {len(stocks)}")

        print("  处理中...")

        all_daily_factors = []

        # 分批处理每只股票
        batch_size = 500
        total_batches = (len(stocks) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_stocks = stocks[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # 获取这些股票的日线数据
            daily_batch = daily[daily['ts_code'].isin(batch_stocks)].copy()

            # 获取这些股票的季度因子
            quarterly_batch = quarterly[quarterly['ts_code'].isin(batch_stocks)].copy()

            if len(quarterly_batch) == 0:
                continue

            # 使用merge_asof进行最近邻匹配
            # 对于每个交易日，找到在它之前（或当天）的最近一次财报公告
            daily_batch = daily_batch.sort_values('trade_date')
            quarterly_batch = quarterly_batch.sort_values('ann_date')

            # 对每只股票进行匹配
            for stock in batch_stocks:
                stock_daily = daily_batch[daily_batch['ts_code'] == stock].copy()
                stock_quarterly = quarterly_batch[quarterly_batch['ts_code'] == stock].copy()

                if len(stock_quarterly) == 0:
                    continue

                # 使用merge_asof
                try:
                    merged = pd.merge_asof(
                        stock_daily,
                        stock_quarterly,
                        left_on='trade_date',
                        right_on='ann_date',
                        by='ts_code',
                        direction='backward'  # 找到在trade_date之前或相等的ann_date
                    )
                    all_daily_factors.append(merged)
                except Exception as e:
                    # 如果merge_asof失败，逐日匹配
                    for _, row in stock_daily.iterrows():
                        trade_date = row['trade_date']
                        valid = stock_quarterly[stock_quarterly['ann_date'] <= trade_date]
                        if len(valid) > 0:
                            factor_row = valid.iloc[-1].to_dict()
                            factor_row['trade_date'] = trade_date
                            all_daily_factors.append(pd.DataFrame([factor_row]))

            if (batch_idx + 1) % 5 == 0:
                print(f"    进度: {min((batch_idx + 1) * batch_size, len(stocks))}/{len(stocks)}")

        if all_daily_factors:
            self.daily_factors = pd.concat(all_daily_factors, ignore_index=True)

            # 清理
            self.daily_factors = self.daily_factors.dropna(subset=['ts_code', 'trade_date'])
            self.daily_factors = self.daily_factors.sort_values(['ts_code', 'trade_date'])

            # 对于在第一次公告之前的记录，使用第一次公告的数据填充
            # 先按ts_code和trade_date排序
            self.daily_factors = self.daily_factors.sort_values(['ts_code', 'trade_date'])

            # 使用forward fill填充ann_date之前的记录
            for stock in self.daily_factors['ts_code'].unique():
                mask = self.daily_factors['ts_code'] == stock
                stock_data = self.daily_factors[mask].copy()

                # 找到第一次公告的因子值
                if 'ann_date' in stock_data.columns:
                    first_valid_idx = stock_data['ann_date'].first_valid_index()
                    if first_valid_idx is not None:
                        first_valid = stock_data.loc[first_valid_idx]

                        # 用第一次公告的值填充之前的记录
                        fill_cols = [c for c in self.factor_cols if c in stock_data.columns]
                        for col in fill_cols:
                            self.daily_factors.loc[mask & (self.daily_factors[col].isna()), col] = first_valid[col]

            print(f"\n  日度因子构建完成: {len(self.daily_factors):,} 条记录")
            print(f"  覆盖日期范围: {self.daily_factors['trade_date'].min()} ~ {self.daily_factors['trade_date'].max()}")

            # 统计
            for col in self.factor_cols[:3]:
                non_null = self.daily_factors[col].notna().sum()
                print(f"    {col}: {non_null:,} 条有效记录")

        return self

    # ==================== 收益率计算 ====================

    def calculate_daily_returns(self, holding_period: int = 20):
        """
        计算未来收益率（基于日度数据）

        Args:
            holding_period: 持有天数，默认20天（约1个月）
        """
        print("\n" + "="*60)
        print(f"计算日度未来收益率 (持有{holding_period}天)")
        print("="*60)

        if self.daily is None:
            print("错误: 无日线数据")
            return self

        df = self.daily.copy()
        df = df.sort_values(['ts_code', 'trade_date'])

        # 计算未来收益率
        df['future_return'] = df.groupby('ts_code')['close'].shift(-holding_period)
        df['future_return'] = (df['future_return'] / df['close']) - 1

        # 转换为日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

        self.returns = df[['ts_code', 'trade_date', 'future_return']].copy()

        print(f"  收益率数据: {len(self.returns):,} 条记录")
        print(f"  收益率统计: 均值={df['future_return'].mean()*100:.2f}%, 标准差={df['future_return'].std()*100:.2f}%")

        return self

    # ==================== IC分析 ====================

    def calculate_ic(self):
        """计算信息系数"""
        print("\n" + "="*60)
        print("因子IC分析 (基于日度因子)")
        print("="*60)

        if self.daily_factors is None or self.returns is None:
            print("错误: 需要先计算日度因子和收益率")
            return None

        # 合并因子和收益
        merged = pd.merge(
            self.daily_factors,
            self.returns,
            on=['ts_code', 'trade_date'],
            how='inner'
        )

        print(f"  合并数据: {len(merged):,} 条记录")

        # 计算每个因子的IC
        ic_results = {}

        for factor in self.factor_cols:
            valid_data = merged[['ts_code', factor, 'future_return']].dropna()

            if len(valid_data) < 100:
                print(f"  {factor}: 数据不足 ({len(valid_data)})")
                continue

            # 计算IC
            ic = valid_data[factor].corr(valid_data['future_return'])

            # t统计量
            n = len(valid_data)
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

            # Rank IC
            spearman_ic, _ = stats.spearmanr(valid_data[factor], valid_data['future_return'])

            ic_results[factor] = {
                'IC': ic,
                'IC_tstat': t_stat,
                'IC_pvalue': p_value,
                'RankIC': spearman_ic,
                'IC_abs': abs(ic),
                'n_samples': len(valid_data)
            }

        ic_df = pd.DataFrame(ic_results).T
        ic_df = ic_df.sort_values('IC_abs', ascending=False)

        print("\n  因子IC分析结果 (按|IC|排序):")
        print("-" * 80)
        print(f"{'因子':<20} {'IC':>10} {'RankIC':>10} {'|IC|':>10} {'P值':>12} {'显著性':>8}")
        print("-" * 80)

        for idx, row in ic_df.iterrows():
            sig = "***" if row['IC_pvalue'] < 0.001 else "**" if row['IC_pvalue'] < 0.01 else "*" if row['IC_pvalue'] < 0.05 else ""
            print(f"{idx:<20} {row['IC']:>10.4f} {row['RankIC']:>10.4f} {row['IC_abs']:>10.4f} {row['IC_pvalue']:>10.4f} {sig:>8}")

        self.ic_results = ic_df

        return ic_df

    # ==================== 分层回测 ====================

    def factor_ranking_analysis(self, factor_col: str, n_groups: int = 5):
        """因子分层回测"""
        merged = pd.merge(
            self.daily_factors,
            self.returns,
            on=['ts_code', 'trade_date'],
            how='inner'
        )

        valid_data = merged[['ts_code', factor_col, 'future_return']].dropna()

        if len(valid_data) < 100:
            return None

        try:
            valid_data['group'] = pd.qcut(valid_data[factor_col], q=n_groups, labels=False, duplicates='drop')
        except Exception as e:
            print(f"  分组失败 ({factor_col}): {e}")
            return None

        group_returns = valid_data.groupby('group').agg({
            'future_return': ['mean', 'std', 'count']
        }).reset_index()
        group_returns.columns = ['group', 'mean_return', 'std_return', 'count']

        long_short_return = group_returns.iloc[-1]['mean_return'] - group_returns.iloc[0]['mean_return']
        factor_direction = '正向' if group_returns['mean_return'].iloc[-1] > group_returns['mean_return'].iloc[0] else '负向'

        return {
            'factor': factor_col,
            'group_returns': group_returns,
            'long_short_return': long_short_return,
            'factor_direction': factor_direction,
            'ic': self.ic_results.loc[factor_col, 'IC'] if factor_col in self.ic_results.index else 0
        }

    def run_all_ranking_analysis(self):
        """运行所有因子的分层分析"""
        print("\n" + "="*60)
        print("因子分层回测分析")
        print("="*60)

        ranking_results = []

        for factor in self.factor_cols:
            result = self.factor_ranking_analysis(factor)
            if result:
                ranking_results.append(result)

        ranking_df = pd.DataFrame([{
            '因子': r['factor'],
            'IC': r['ic'],
            '多空收益': r['long_short_return'] * 100,
            '方向': r['factor_direction'],
            'Q1(低)收益': r['group_returns']['mean_return'].iloc[0] * 100,
            'Q5(高)收益': r['group_returns']['mean_return'].iloc[-1] * 100
        } for r in ranking_results])

        ranking_df = ranking_df.sort_values('多空收益', ascending=False)

        print("\n  分层回测结果 (按多空收益排序):")
        print("-" * 90)
        print(f"{'因子':<20} {'IC':>8} {'多空收益':>12} {'方向':>8} {'Q1收益':>10} {'Q5收益':>10}")
        print("-" * 90)

        for _, row in ranking_df.iterrows():
            print(f"{row['因子']:<20} {row['IC']:>8.4f} {row['多空收益']:>10.2f}% {row['方向']:>8} {row['Q1(低)收益']:>8.2f}% {row['Q5(高)收益']:>8.2f}%")

        self.ranking_results = ranking_df

        return ranking_df

    # ==================== 可视化 ====================

    def visualize_results(self):
        """可视化分析结果"""
        if self.ic_results is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. IC柱状图
        ax1 = axes[0, 0]
        ic_sorted = self.ic_results.sort_values('IC')
        colors = ['green' if x > 0 else 'red' for x in ic_sorted['IC']]
        ax1.barh(range(len(ic_sorted)), ic_sorted['IC'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(ic_sorted)))
        ax1.set_yticklabels(ic_sorted.index, fontsize=8)
        ax1.set_xlabel('IC')
        ax1.set_title('Factor Information Coefficient (IC)')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(x=0.02, color='green', linestyle='--', alpha=0.5)

        # 2. |IC|排名
        ax2 = axes[0, 1]
        ic_abs = self.ic_results.sort_values('IC_abs', ascending=True)
        ax2.barh(range(len(ic_abs)), ic_abs['IC_abs'], color='steelblue', alpha=0.7)
        ax2.set_yticks(range(len(ic_abs)))
        ax2.set_yticklabels(ic_abs.index, fontsize=8)
        ax2.set_xlabel('|IC|')
        ax2.set_title('Factor |IC| Ranking (Predictive Power)')
        ax2.axvline(x=0.02, color='red', linestyle='--', label='Weak (0.02)')
        ax2.axvline(x=0.05, color='orange', linestyle='--', label='Medium (0.05)')
        ax2.legend(fontsize=8)

        # 3. 分层收益 (Top 5)
        ax3 = axes[1, 0]
        if hasattr(self, 'ranking_results') and self.ranking_results is not None:
            top5 = self.ranking_results.head(5)
            x = range(len(top5))
            ax3.bar(x, top5['多空收益'], color='steelblue', alpha=0.7)
            ax3.set_xticks(x)
            ax3.set_xticklabels(top5['因子'], rotation=45, ha='right')
            ax3.set_ylabel('Long-Short Return (%)')
            ax3.set_title('Top 5 Factors by Long-Short Return')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 4. 显著性统计
        ax4 = axes[1, 1]
        sig_counts = {
            '*** (p<0.001)': (self.ic_results['IC_pvalue'] < 0.001).sum(),
            '** (p<0.01)': ((self.ic_results['IC_pvalue'] >= 0.001) & (self.ic_results['IC_pvalue'] < 0.01)).sum(),
            '* (p<0.05)': ((self.ic_results['IC_pvalue'] >= 0.01) & (self.ic_results['IC_pvalue'] < 0.05)).sum(),
            'Not Sig.': (self.ic_results['IC_pvalue'] >= 0.05).sum()
        }
        colors = ['green', 'lightgreen', 'yellow', 'red']
        ax4.bar(sig_counts.keys(), sig_counts.values(), color=colors, alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Statistical Significance Distribution')
        for i, (k, v) in enumerate(sig_counts.items()):
            ax4.text(i, v + 0.1, str(v), ha='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'ic_analysis_v2.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n分析图已保存: {self.output_dir / 'ic_analysis_v2.png'}")

    # ==================== 保存结果 ====================

    def save_results(self):
        """保存分析结果"""
        print("\n" + "="*60)
        print("保存分析结果")
        print("="*60)

        if self.daily_factors is not None:
            self.daily_factors.to_parquet(self.output_dir / 'daily_factors.parquet', index=False)
            print(f"  日度因子数据: {self.output_dir / 'daily_factors.parquet'}")

        if self.ic_results is not None:
            self.ic_results.to_csv(self.output_dir / 'ic_results_v2.csv')
            print(f"  IC结果: {self.output_dir / 'ic_results_v2.csv'}")

        if self.ranking_results is not None:
            self.ranking_results.to_csv(self.output_dir / 'ranking_results_v2.csv', index=False)
            print(f"  分层结果: {self.output_dir / 'ranking_results_v2.csv'}")

    # ==================== 完整分析流程 ====================

    def run_full_analysis(self, start_period='20240331', end_period='20241231',
                         start_day='20250101', end_day='20260206',
                         holding_period=20):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print("中证1000基本面因子分析V2 - 日度因子版")
        print("="*80)
        print(f"分析参数: 持有期={holding_period}天, 财报期间={start_period}~{end_period}")

        # 1. 加载数据
        self.load_financial_data(start_period, end_period)
        self.load_daily_data(start_day, end_day)

        # 2. 合并财务数据
        self.merge_financial_data()

        # 3. 构建季度因子
        self.build_quarterly_factors()

        # 4. 转换为日度因子（核心改进）
        self.convert_to_daily_factors()

        # 5. 计算收益率
        self.calculate_daily_returns(holding_period=holding_period)

        # 6. IC分析
        self.calculate_ic()

        # 7. 分层回测
        self.run_all_ranking_analysis()

        # 8. 可视化
        self.visualize_results()

        # 9. 保存结果
        self.save_results()

        print("\n" + "="*80)
        print("分析完成!")
        print("="*80)

        return self


# ==================== 主程序 ====================

if __name__ == "__main__":
    analyzer = FundamentalFactorAnalyzerV2(output_dir="./fundamental_analysis_results_v2")
    analyzer.run_full_analysis(holding_period=20)
